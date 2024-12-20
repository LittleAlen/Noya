#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <regex>
#include <omp.h>
#include <cmath>
#include <cassert>

#include "seal/seal.h"
#include "Util.h"
#include "SealBfvVector.h"
using namespace std;
using namespace seal;

namespace mycryptonets
{
    struct AtomicSealBfvEnvironment
    {
        /*
        * Parameter selections
        * First pick a good plaintext modulus so that computation won't overflow
        * Depth is roughly determined by log2(Q/t) - 1. For more depth, increase Q (coeff_modulus). 
        *    - Fine tune Q by adopting trial-and-error
        *    - Specifically, start from a modestly small Q, evaluate the circuit. If it fails, then increase Q and so on.
        * Choose n based on Q according to the expected security level.
        *    - There's a map from homomorphicencryption.org
        */
        AtomicSealBfvEnvironment() = default;

        AtomicSealBfvEnvironment(size_t poly_modulus_degree, uint64_t plain_modulus)
        {
            EncryptionParameters parms(scheme_type::BFV);
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
            // auto k=CoeffModulus::BFVDefault(poly_modulus_degree);
            // for (int i=0;i<k.size();i++){
            //     cout<<"coeffModulus  "<<k[i]<<endl;
            // }
            
            parms.set_plain_modulus(plain_modulus);

            context = SEALContext::Create(parms);
            KeyGenerator keygen(context);
            public_key = keygen.public_key();
            secret_key = keygen.secret_key();
            relin_keys = keygen.relin_keys_local();
            gal_keys = keygen.galois_keys_local();

            encryptorPtr = make_shared<Encryptor>(context, public_key);
            decryptorPtr = make_shared<Decryptor>(context, secret_key);
            batchEncoderPtr = make_shared<BatchEncoder>(context);
            evaluatorPtr = make_shared<Evaluator>(context);
        }
        ~AtomicSealBfvEnvironment() {}

        shared_ptr<SEALContext> context;
        PublicKey public_key;
        SecretKey secret_key;
        RelinKeys relin_keys;
        GaloisKeys gal_keys;

        shared_ptr<Encryptor> encryptorPtr;
        shared_ptr<Decryptor> decryptorPtr;
        shared_ptr<Evaluator> evaluatorPtr;
        shared_ptr<BatchEncoder> batchEncoderPtr;
    };

    struct SealBfvEnvironment
    {
        SealBfvEnvironment() = default;
        SealBfvEnvironment(size_t poly_modulus_degree, vector<Modulus> plain_modulus)
            : poly_modulus_degree(poly_modulus_degree)
        {
            vector<uint64_t> plain_modulus_uint64;
            transform(plain_modulus.begin(),
                      plain_modulus.end(),
                      back_inserter(plain_modulus_uint64),
                      [](const Modulus &modulus) { return modulus.value(); });
            set(poly_modulus_degree, plain_modulus_uint64);
        }
        SealBfvEnvironment(size_t poly_modulus_degree, vector<uint64_t> plain_modulus)
            : poly_modulus_degree(poly_modulus_degree)
        {
            set(poly_modulus_degree, plain_modulus);
        }
        ~SealBfvEnvironment() {}

        void set(size_t poly_modulus_degree, vector<uint64_t> plain_modulus)
        {
            uIntBigFactor = BigUInt("1");
            // int num=uIntBigFactor.bit_count();
            // cout<<num<<endl;
            for (auto plain_modulo : plain_modulus)
            {   
                cout<<"Plaintext Modulus: "<<plain_modulo<<endl;
                environments.emplace_back(poly_modulus_degree, plain_modulo);
                auto factor = to_hex(plain_modulo);
                uIntBigFactor *= factor;
                factors.emplace_back(move(factor));
            }
            for (auto &factor : factors)//it seems to be used to compute the coefficients modules
            {
                auto minor = uIntBigFactor / factor;
                BigUInt rem;
                minor.divrem(factor, rem);
                auto ys = rem.modinv(factor);
                preComputedCoefficients.emplace_back(ys * minor);
            }
        }

        size_t poly_modulus_degree;
        vector<BigUInt> factors;//all the plain_modulus
        vector<AtomicSealBfvEnvironment> environments;
        BigUInt uIntBigFactor;// deriving from plain_modulus  * 
        vector<BigUInt> preComputedCoefficients;//deriving from plain_modulus corresponding
    };

    vector<uint64_t> splitBigNumbers(
        double num, double scale, const SealBfvEnvironment &env)//computing the scaled value among different plain_modules
    {
        size_t envCount = env.environments.size();
        vector<uint64_t> res(envCount, 0);

        auto w = (long long)round(num * scale);
        BigUInt z = w < 0 ? env.uIntBigFactor - (uint64_t)abs(w) : BigUInt(to_hex(w));

        for (size_t i = 0; i < envCount; i++)
        {
            BigUInt temp;
            z.divrem(env.factors[i], temp);
            res[i] = (uint64_t)temp.to_double();
        }

        return res;
    }
     vector<uint64_t> splitBigNumbers(
        int num, double scale, const SealBfvEnvironment &env)//computing the scaled value among different plain_modules
    {
        size_t envCount = env.environments.size();
        vector<uint64_t> res(envCount, 0);

        auto w = (long long)round(num * scale);
        BigUInt z = w < 0 ? env.uIntBigFactor - (uint64_t)abs(w) : BigUInt(to_hex(w));

        for (size_t i = 0; i < envCount; i++)
        {
            BigUInt temp;
            z.divrem(env.factors[i], temp);
            res[i] = (uint64_t)temp.to_double();
        }

        return res;
    }

    double joinSplitNumbers(const vector<uint64_t> &split, const SealBfvEnvironment &env)
    {
        BigUInt res("0");
        for (size_t i = 0; i < env.environments.size(); i++)
        {
            res += env.preComputedCoefficients[i] * split[i];
        }
        res.divrem(env.uIntBigFactor, res);
        if (res * 2 > env.uIntBigFactor)
        {
            return -1 * (env.uIntBigFactor - res).to_double();
        }
        return res.to_double();
    }

    // A wrapper class for Ciphertext with CRT
    struct SealBfvCiphertext
    {
        
        SealBfvCiphertext(const vector<double> &m,
                          const SealBfvEnvironment &env,
                          size_t scale = 1) : batchSize(m.size()), scale(scale)
        {
            size_t envCount = env.environments.size();
            assert(envCount > 0);
            vector<vector<uint64_t>> split(envCount, vector<uint64_t>(batchSize, 0));
            for (size_t i = 0; i < batchSize; i++)
            {
                auto temp = splitBigNumbers(m[i], scale, env);
                for (size_t j = 0; j < envCount; j++)
                {
                    split[j][i] = temp[j];//将这个数组每个元素通过CRT方式，计算多个系数模数对应的值，横坐标代表，多个系数模数
                }
            }
            for (size_t j = 0; j < envCount; j++)
            {
                Plaintext temp_p;
                env.environments[j].batchEncoderPtr->encode(split[j], temp_p);
                Ciphertext temp_c;
                env.environments[j].encryptorPtr->encrypt(move(temp_p), temp_c);
                eVectors.emplace_back(move(temp_c));
            }
        }

        SealBfvCiphertext(const SealBfvCiphertext &ciphertext) : batchSize(ciphertext.batchSize),
                                                                 scale(ciphertext.scale),
                                                                 eVectors(ciphertext.eVectors)
        {
        }

        SealBfvCiphertext(size_t scale=1): batchSize(0), scale(scale)
        {
        
        }

        SealBfvCiphertext &operator=(const SealBfvCiphertext &ciphertext)
        {
            batchSize = ciphertext.batchSize;
            scale = ciphertext.scale;
            eVectors = ciphertext.eVectors;
            return *this;
        }

        ~SealBfvCiphertext() {}

        vector<double> decrypt(const SealBfvEnvironment &env) const
        {

            size_t envCount = eVectors.size();
            vector<vector<uint64_t>> split(batchSize, vector<uint64_t>(envCount, 0));

            for (size_t j = 0; j < envCount; j++)
            {
                Plaintext temp_p;
                env.environments[j].decryptorPtr->decrypt(eVectors[j], temp_p);
                vector<uint64_t> temp_vec;
                env.environments[j].batchEncoderPtr->decode(move(temp_p), temp_vec);
                for (size_t i = 0; i < batchSize; i++)
                {
                    split[i][j] = temp_vec[i];
                }
            }

            vector<double> res(batchSize, 0.0);
            for (size_t i = 0; i < batchSize; i++)
            {
                res[i] = joinSplitNumbers(split[i], env) / scale;
            }
            return res;
        }

        vector<Ciphertext> eVectors;
        size_t scale;
        // Set bits from left to right
        size_t batchSize;
    };

    // A wrapper class for Plaintext with CRT
    struct SealBfvPlaintext
    {
        SealBfvPlaintext() = default;

        SealBfvPlaintext(double m,
                         const SealBfvEnvironment &env,
                         size_t scale = 1) : batchSize(env.poly_modulus_degree), scale(scale)
        {
            vector<uint64_t> split = splitBigNumbers(m, scale, env);
            for (auto num : split)
            {
                pVectors.emplace_back(to_hex(num));
            }
        }

        SealBfvPlaintext(const vector<double> &m,
                         const SealBfvEnvironment &env,
                         size_t scale = 1) : batchSize(m.size()), scale(scale)
        {
            size_t envCount = env.environments.size();
            assert(envCount > 0);
            vector<vector<uint64_t>> split(envCount, vector<uint64_t>(batchSize, 0));
            for (size_t i = 0; i < batchSize; i++)
            {
                auto temp = splitBigNumbers(m[i], scale, env);
                for (size_t j = 0; j < envCount; j++)
                {
                    split[j][i] = temp[j];
                }
            }
            for (size_t j = 0; j < envCount; j++)
            {
                Plaintext temp_p;
                env.environments[j].batchEncoderPtr->encode(split[j], temp_p);
                pVectors.emplace_back(move(temp_p));
            }
        }
        SealBfvPlaintext(const vector<int> &m,
                         const SealBfvEnvironment &env,
                         int scale = 1) : batchSize(m.size()), scale(scale)
        {
            size_t envCount = env.environments.size();
            assert(envCount > 0);
            vector<vector<uint64_t>> split(envCount, vector<uint64_t>(batchSize, 0));
            for (size_t i = 0; i < batchSize; i++)
            {
                auto temp = splitBigNumbers(m[i], scale, env);
                for (size_t j = 0; j < envCount; j++)
                {
                    split[j][i] = temp[j];
                }
            }
            for (size_t j = 0; j < envCount; j++)
            {
                Plaintext temp_p;
                env.environments[j].batchEncoderPtr->encode(split[j], temp_p);
                pVectors.emplace_back(move(temp_p));
            }
        }
        ~SealBfvPlaintext() {}

        bool is_zero() const
        {
            return all_of(pVectors.begin(),
                          pVectors.end(),
                          [](const Plaintext &p) { return p.is_zero(); });
        }

        vector<double> decode(const SealBfvEnvironment &env) const
        {

            size_t envCount = pVectors.size();
            vector<vector<uint64_t>> split(batchSize, vector<uint64_t>(envCount, 0));

            for (size_t j = 0; j < envCount; j++)
            {
                
                vector<uint64_t> temp_vec;
                env.environments[j].batchEncoderPtr->decode(move(pVectors[j]), temp_vec);
                for (size_t i = 0; i < batchSize; i++)
                {
                    split[i][j] = temp_vec[i];
                }
            }

            vector<double> res(batchSize, 0.0);
            for (size_t i = 0; i < batchSize; i++)
            {
                res[i] = joinSplitNumbers(split[i], env) / scale;
            }
            return res;
        }
        vector<Plaintext> pVectors;
        size_t scale;
        size_t batchSize;
    };


} // namespace mycryptonets