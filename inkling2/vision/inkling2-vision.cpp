// inkling2-vision : l'encodeur d'image d'Inkling 2 Mini (mmproj de llama.cpp), seul, sans le modèle de texte.
//
//   inkling2-vision --mmproj F [--max-tokens N] [--threads N] IMAGE SORTIE     une image, puis quitte
//   inkling2-vision --mmproj F [--max-tokens N] --serve          une ligne « IMAGE<TAB>SORTIE » par image sur stdin,
//                                                                réponse « OK <rangs> <largeur> <hauteur> <ms> » ou « ERR … »
//
// SORTIE : « INKV » (4 octets), puis int32 rangs, int32 dimension (2304), puis rangs x dimension float32. Ce sont les
// vecteurs projetés que llama.cpp met à la place du token image (<|unused_200054|>), sans la norme des embeddings
// (inkling2.cpp, build_inputs). Le moteur les charge par « IMG SORTIE ».
//
// Il faut mtmd compilé avec mtmd-sans-vocab.patch : sans modèle de texte, mtmd n'a pas de vocabulaire, et le marqueur
// <|content_image|> qu'il ajoute autour de l'image est ignoré (la façade l'a déjà mis, par le gabarit officiel).
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

static constexpr int H = 2304;   // text_hidden_size d'Inkling 2 Mini

static std::string encode(mtmd_context * ctx, const std::string & image, const std::string & out, int & rows, int & nx,
                          int & ny, double & ms) {
    const auto t0 = std::chrono::steady_clock::now();
    mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(ctx, image.c_str(), false, mtmd_helper_init_opt_default());
    if (!w.bitmap) return "image illisible : " + image;
    nx = (int) mtmd_bitmap_get_nx(w.bitmap);
    ny = (int) mtmd_bitmap_get_ny(w.bitmap);
    mtmd_input_part part{nullptr, w.bitmap};
    const mtmd_input_part * parts[1] = {&part};
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    std::string err;
    if (mtmd_tokenize_from_parts(ctx, chunks, parts, 1, false) != 0) err = "prétraitement de l'image refusé";
    const mtmd_input_chunk * img = nullptr;
    for (size_t i = 0; err.empty() && i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            if (img) err = "plusieurs morceaux d'image (découpage inattendu)";
            img = c;
        }
    }
    if (err.empty() && !img) err = "aucun morceau d'image";
    if (err.empty()) {
        rows = (int) mtmd_input_chunk_get_n_tokens(img);
        if (mtmd_input_chunk_get_n_pos(img) != rows) err = "positions et rangs différents (attendu : une position par rang)";
        else if (mtmd_encode_chunk(ctx, img) != 0) err = "encodage refusé";
    }
    if (err.empty()) {
        const float * e = mtmd_get_output_embd(ctx);
        FILE * f = std::fopen(out.c_str(), "wb");
        if (!f) err = "écriture impossible : " + out;
        else {
            const int32_t head[2] = {rows, H};
            const bool ok = std::fwrite("INKV", 1, 4, f) == 4 && std::fwrite(head, 4, 2, f) == 2 &&
                            std::fwrite(e, 4, (size_t) rows * H, f) == (size_t) rows * H;
            if (std::fclose(f) != 0 || !ok) err = "écriture incomplète : " + out;
        }
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(w.bitmap);
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return err;
}

int main(int argc, char ** argv) {
    std::string mmproj;
    bool serve = false;
    int max_tokens = -1;
    int threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);   // sur le processeur (encodeur CPU)
    std::string pos[2];
    int npos = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mmproj" && i + 1 < argc) mmproj = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--serve") serve = true;
        else if (npos < 2) pos[npos++] = a;
    }
    if (mmproj.empty() || (!serve && npos != 2)) {
        std::fprintf(stderr, "usage : %s --mmproj F [--max-tokens N] (IMAGE SORTIE | --serve)\n", argv[0]);
        return 2;
    }
    mtmd_context_params p = mtmd_context_params_default();
    p.print_timings = false;
    p.warmup = false;
    p.image_max_tokens = max_tokens;
    p.n_threads = threads;
    mtmd_context * ctx = mtmd_init_from_file(mmproj.c_str(), nullptr, p);
    if (!ctx) { std::fprintf(stderr, "mmproj illisible : %s\n", mmproj.c_str()); return 1; }
    if (!mtmd_support_vision(ctx)) { std::fprintf(stderr, "ce mmproj n'a pas d'encodeur d'image\n"); return 1; }
    int rows = 0, nx = 0, ny = 0;
    double ms = 0;
    if (!serve) {
        const std::string err = encode(ctx, pos[0], pos[1], rows, nx, ny, ms);
        mtmd_free(ctx);
        if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("%d rangs (%dx%d), %.0f ms\n", rows, nx, ny, ms);
        return 0;
    }
    std::printf("READY\n");
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "QUIT") break;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) { std::printf("ERR attendu : IMAGE<TAB>SORTIE\n"); std::fflush(stdout); continue; }
        const std::string err = encode(ctx, line.substr(0, tab), line.substr(tab + 1), rows, nx, ny, ms);
        if (err.empty()) std::printf("OK %d %d %d %.0f\n", rows, nx, ny, ms);
        else std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
    }
    mtmd_free(ctx);
    return 0;
}
