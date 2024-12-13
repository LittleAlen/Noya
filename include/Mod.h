#pragma once

#include <tuple>

using namespace std;

namespace mycryptonets
{
    

    // 扩展欧几里得算法
    std::tuple< long long, long long, long long> extendedGCD( long long a, long long b) {
        if (b == 0) {
            return std::make_tuple(a, 1, 0);
        } else {
            auto [g, x, y] = extendedGCD(b, a % b);
            return std::make_tuple(g, y, x - (a / b) * y);
        }
    }

    // 求模逆
    long long modInverse( long long modulus, unsigned long long value) {
        auto [g, x, y] = extendedGCD(value, modulus);
        if (g != 1) {
            // 逆不存在
            throw "Modular inverse does not exist";
        } else {
            // 由于结果可能为负，我们需要将结果加上模数来确保结果为正
            return (x % modulus + modulus) % modulus;
        }
    }
} // namespace mycryptonets
