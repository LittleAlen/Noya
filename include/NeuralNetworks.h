#pragma once

#include <math.h>

#include "SealBfvVector.h"

namespace mycryptonets
{
    void fc(vector<SealBfvCiphertext const *> inputPtr,//Conv 13 X 13 X 25     FC1 845X1
            vector<SealBfvPlaintext> &weights,//Conv 25 X 5       FC1 845X845
            vector<SealBfvPlaintext> &biases,// Conv 5 X 1       FC1 845X1
            size_t dotLen,//Conv 25              FC1 845
            vector<SealBfvCiphertext> &destination,
            const SealBfvEnvironment &env)
    {
        assert((inputPtr.size() * weights.size()) % (dotLen * dotLen) == 0);
        destination = vector<SealBfvCiphertext>(
            (inputPtr.size() * weights.size()) / (dotLen * dotLen),
            SealBfvCiphertext(biases[0].scale));

        // for(int i=0;i<(inputPtr.size() * weights.size()) / (dotLen * dotLen);i++){
        //     destination.emplace_back(move(SealBfvCiphertext(biases[0].scale)));
        // }

        // cout<<"idx:   "<<0<<endl;
        // cout<<" "<<destination[0].batchSize<<" "<<destination[0].scale<<"  "<<destination[0].eVectors.size()<<endl;

        // cout<<"idx:   "<<1<<endl;
        // cout<<" "<<destination[1].batchSize<<" "<<destination[1].scale<<"  "<<destination[1].eVectors.size()<<endl;

        // cout<<"idx:   "<<2<<endl;
        // cout<<" "<<destination[2].batchSize<<" "<<destination[2].scale<<"  "<<destination[2].eVectors.size()<<endl;

#pragma omp parallel for
        for (size_t i = 0; i < weights.size(); i += dotLen)//start
        {
#pragma omp parallel for
            for (size_t j = 0; j < inputPtr.size(); j += dotLen)
            {
                vector<SealBfvCiphertext> dots;
#pragma omp parallel for                
                for (size_t x = 0; x < dotLen; x++)
                {
                    SealBfvCiphertext temp(biases[0].scale);
                    if (weights[i + x].is_zero() || inputPtr[j + x] == nullptr)
                    {
                        continue;
                    }
                    multiply_plain(*inputPtr[j + x], weights[i + x], temp, env);
                    dots.emplace_back(move(temp));
                }
                size_t idx = (i / dotLen) * inputPtr.size() / dotLen + (j / dotLen);
                if (dots.size() > 0)
                {
                    add_many(dots, destination[idx], env);
                }
                // else
                // {   
                    //  cout<<"      come"<<endl;
                    // Put an encrypted 0
                    // destination[idx] = SealBfvCiphertext(vector<double>(env.poly_modulus_degree, 0),//CKKS 方案时要除以2
                    //                                      env,
                    //                                      inputPtr[0]->scale * weights[0].scale);
                    
                // }
                add_plain_inplace(destination[idx], biases[i / dotLen], env);
            }
        }
    }

    template <typename T>
    void convolutionOrganizer(
        const vector<T> &data,
        size_t kernelDim,//5
        size_t stride,//2
        size_t padding, //1 to the right bottom direction
        vector<T const *> &dataPTr)
    {
        size_t dim = (size_t)sqrt(data.size());
        size_t outputDim = (dim + padding - kernelDim) / stride + 1;
        size_t kernelSize = kernelDim * kernelDim;
        dataPTr = vector<T const *>(outputDim * outputDim * kernelSize, nullptr);

        #pragma omp parallel for
        for (size_t i = 0; i < outputDim; i++)
        {
            for (size_t j = 0; j < outputDim; j++)
            {
                size_t start = ((i * outputDim) + j) * kernelSize;
                for (size_t x = 0; x < kernelDim; x++)
                {
                    for (size_t y = 0; y < kernelDim; y++)
                    {
                        size_t row = i * stride + x;
                        size_t col = j * stride + y;
                        dataPTr[start + x * kernelDim + y] =
                            row < dim && col < dim ? &data[row * dim + col] : nullptr;
                    }
                }
            }
        }
    }
} // namespace mycryptonets