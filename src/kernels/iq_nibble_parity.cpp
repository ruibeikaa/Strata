// src/kernels/iq_nibble_parity.cpp - P1 part D (K1), M4: the decode path on 4-bit slots against the production
// kernels, bit for bit (D_DESIGN.md section 8(a)).
//
//     build/iq_nibble_parity --header    the host half of strata/core/expert_nibble.hpp (no GPU)
//
// To come: the transcoder's round trip on every item (real experts of both packs plus synthetic blocks: every grid
// index, sign byte and scale, d at the fp16 extremes), the device slot hash against `hash_host`, and the whole
// native_expert_grouped on converted slots against the GGUF blobs (gate, up, h, hq and out byte for byte; T = 1..6,
// 0..11 entries a group, G = 1..cap, both grid strides; R4S, R4P, R4SV, R4PX and the chosen table).
//
// Until M4 only `--header` runs: the item code's encode / decode round trip over every value each format can hold,
// the layout's sizes and the variant table.  Without it the program reports itself skipped (77).
#include "strata/core/expert_nibble.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace nib = strata::core::nibble;

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}

// Every value the format holds at every one of an item's 32 positions (the other positions cycle through the set),
// encoded, decoded and compared; and one value outside the set must be refused.
void round_trip(int fmt, const std::vector<int8_t>& set, int8_t outside) {
    int8_t g[32], back[32];
    uint32_t q[4];
    for (size_t shift = 0; shift < set.size(); ++shift) {
        for (int i = 0; i < 32; ++i) g[i] = set[(shift + (size_t) i) % set.size()];
        const bool ok = nib::encode_item(fmt, g, q);
        nib::decode_item(fmt, q, back);
        expect(ok && std::memcmp(g, back, 32) == 0, (std::string(nib::fmt_name(fmt)) + " round trip").c_str());
    }
    for (int i = 0; i < 32; ++i) g[i] = set[0];
    g[17] = outside;
    expect(!nib::encode_item(fmt, g, q), (std::string(nib::fmt_name(fmt)) + " refuses a value outside the set").c_str());
}

int header_check() {
    expect(nib::kGuBytes == 1766400 && nib::kScOff == 1638400 && nib::kDOff == 1740800, "layout sizes");
    expect(nib::code_off(nib::kNFf, 0) == nib::kUpCodesOff, "up rows' codes");
    expect(nib::slot_bytes_for(921600) % nib::kSlotAlign == 0, "slot alignment");
    round_trip(nib::kIQ2_S, { 8, 25, 43, -8, -25, -43 }, 9);
    round_trip(nib::kIQ3_XXS, { 4, 12, 20, 28, 36, 44, 52, 62, -4, -12, -20, -28, -36, -44, -52, -62 }, 5);
    std::vector<int8_t> s3;
    for (int v = -15; v <= 15; v += 2) s3.push_back((int8_t) v);
    round_trip(nib::kIQ3_S, s3, 2);
    // IQ2_S's nibble is the WS kernel's PRMT selector into {8,25,43,0} {-8,-25,-43,0}
    {
        int8_t g[32] = {}, back[32];
        uint32_t q[4];
        for (int i = 0; i < 32; ++i) g[i] = (int8_t) (i % 2 ? -25 : 43);
        nib::encode_item(nib::kIQ2_S, g, q);
        expect((q[0] & 0xFu) == 2u && (q[0] >> 4 & 0xFu) == 5u, "IQ2_S selector nibbles");
        nib::decode_item(nib::kIQ2_S, q, back);
        expect(std::memcmp(g, back, 32) == 0, "IQ2_S selector decode");
    }
    expect(nib::gu_variants(nib::kIQ3_S).single == nib::GuVariant::kR4SV &&
               nib::gu_variants(nib::kIQ3_S).shared == nib::GuVariant::kR4PX &&
               nib::gu_variants(nib::kIQ2_S).single == nib::GuVariant::kR4P &&
               nib::gu_variants(nib::kIQ3_XXS).shared == nib::GuVariant::kR4S &&
               nib::gu_variants(23).single == nib::GuVariant::kProduction,
           "variant table");
    {
        const uint8_t b[6] = { 1, 2, 3, 4, 5, 6 };
        const uint64_t h = nib::hash_host(b, 6);
        const uint64_t want = (uint64_t) (0x04030201u ^ 0u) * nib::kHashMul + (uint64_t) (0x00000605u ^ 1u) * nib::kHashMul;
        expect(h == want, "hash_host");
    }
    std::printf("%s (%d failures)\n", fails == 0 ? "header OK" : "header FAILED", fails);
    return fails == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--header") == 0) return header_check();
    if (argc != 1) {
        std::fprintf(stderr, "usage: iq_nibble_parity [--header]\n");
        return 2;
    }
    std::printf("iq_nibble_parity: the GPU parity is not implemented yet (P1 part D, M4) - skipped; --header checks the header\n");
    return 77;
}
