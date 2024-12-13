#include<stdlib.h>
#include<time.h>
#include<fstream>
#include <sstream>
#include<iostream>
#include "NeuralNetworks.h"
#include "MnistDataPreprocess.h"
#include "SRHE.h"
#include "Mod.h"
using namespace std;
using namespace seal;
using namespace mycryptonets;
/*
I use the same network and parameter, however, it runs out of noise budget, so I decrypt output before delivering to next layer in some layers,
the result is 98.69% which is close to 98.95% 
*/
namespace {
    size_t POLY_MODULUS_DEGREE = 8192; 
    size_t RemoveScale=512;
    size_t ReverseScale=0;
    //Origin
    // size_t p1=549764251649;
    // size_t p2=549764284417;
    //  vector<Modulus> PLAINTEXT_MODULUS = {p1,p2};
    // size_t INPUT_SCALE = 16;
    // size_t CONV_SCALE = 32;
    // size_t FC1_SCALE = 1024;
    // size_t FC2_SCALE = 32;
    vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
        POLY_MODULUS_DEGREE, {18, 18});
    //square
    // size_t INPUT_SCALE = 2;
    // size_t CONV_SCALE = 2;
    // size_t FC1_SCALE = 32;
    // size_t FC2_SCALE = 16;
    //Relu
    size_t INPUT_SCALE = 8;
    size_t CONV_SCALE = 32;
    size_t FC1_SCALE = 64*2;
    size_t FC2_SCALE = 64*2;
}
//4 93.2007


vector<SealBfvCiphertext> Cryptonets(const Params &params, const vector<SealBfvCiphertext> &inputE, const SealBfvEnvironment &env)
{
    //inputE 784 X sealbfvciphertext 
    vector<SealBfvCiphertext> intermediateResultsE;

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    // Conv Layer
    {
        vector<SealBfvCiphertext> resultE;
        auto decor = make_decorator(fc, "Conv");

        vector<SealBfvCiphertext const *> inputEPtr;
        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;

        convolutionOrganizer(inputE, 5, 2, 1, inputEPtr);//13 X 13 X 25
        encode(params.convWeights, WeightsP, env, CONV_SCALE);//scale =2  Matrix: 25X5 
        encode(params.convBiases, BiasesP, env, inputE[0].scale * CONV_SCALE); //scale=4  Matrix: 5X1
        decor(inputEPtr,
              WeightsP,
              BiasesP,
              5 * 5, // 重合部分
              resultE,
              env);//Matrix： 13X13X5

        intermediateResultsE = move(resultE);
        cout<<"-Scale :"<<intermediateResultsE[0].scale<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }

    // // Modulus switch
    // {
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < intermediateResultsE.size(); i++) {
    //         mod_switch_to_next_inplace(intermediateResultsE[i], env);
    //     }
    // }
    RemoveScale=intermediateResultsE[0].scale;
    ReverseScale=modInverse((size_t)147457*(size_t)163841,RemoveScale);
  
    // // // Square activation layer  scale=16
    {
        auto decor = make_decorator(square_inplace_vec, "Square");

        decor(intermediateResultsE, env);
        cout<<"-Scale :"<<intermediateResultsE[0].scale<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }  
    //Removeing Scale
    // {
    //     SealBfvPlaintext tmp(ReverseScale,env,1);
    //     for(int i=0;i<intermediateResultsE.size();i++){
    //         multiply_plain(intermediateResultsE[i],tmp,intermediateResultsE[i],env);
    //         intermediateResultsE[i].scale/=RemoveScale;
    //     }
    // }
    //Debug Only
    // vector<vector<double>> res;
    // decrypt(intermediateResultsE,res,env);
    // encrypt(res,intermediateResultsE,env,intermediateResultsE[0].scale);


    // replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
   // simulate_relu(intermediateResultsE,env);

    // Debug ReLU
    // {
    //     vector<vector<double>>res;
    //     size_t scale=intermediateResultsE[0].scale;
    //     decrypt(intermediateResultsE, res, env);
    //     for(int i=0;i<res.size();i++){
    //         for(int j=0;j<res[i].size();j++){
    //             if(res[i][j]<0){
    //                 res[i][j]=0.0;
    //             }
    //         }
    //     }
    //     vector<SealBfvCiphertext> resultE;
    //     for(int i=0;i<res.size();i++)
    //     {
    //         resultE.emplace_back(move(SealBfvCiphertext(res[i],env,scale)));
    //     }
    //     intermediateResultsE=move(resultE);
    // }


    // FC1 Layer
    {
        vector<SealBfvCiphertext> resultE;
        auto decor = make_decorator(fc, "FC1");

        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;

        encode(params.FC1Weights, WeightsP, env, FC1_SCALE);//Matrix 13X13X5X845 全卷积 
       
        encode(params.FC1Biases, BiasesP, env, intermediateResultsE[0].scale * FC1_SCALE);//scale = 16*32  Matrix: 845X1
        decor(getPointers(intermediateResultsE),
              WeightsP,
              BiasesP,
              intermediateResultsE.size(),
              resultE,
              env);//matrix 845X1

        intermediateResultsE.clear();
        intermediateResultsE = move(resultE); 
        cout<<"-Scale :"<<intermediateResultsE[0].scale<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
  
    
    
    
    // RemoveScale=2;
    // ReverseScale=modInverse((size_t)147457*(size_t)163841,RemoveScale);
    // //ReverseScale=((unsigned long long )147457*(unsigned long long)163841+1)/2;
    // cout<<"Reverse Scale  "<<ReverseScale<<endl;
    // // Square activation layer
    {
        auto decor = make_decorator(square_inplace_vec, "Square");

        decor(intermediateResultsE, env);
        cout<<"-Scale :"<<intermediateResultsE[0].scale<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
   //Removeing Scale
    // {
    //     vector<vector<double>> res1;
    //     decrypt(intermediateResultsE,res1,env);
    //     cout<<"-------------------Before "<<res1[0][0]<<endl;
    //     SealBfvPlaintext tmp(ReverseScale,env,1);
    //     for(int i=0;i<intermediateResultsE.size();i++){
    //         multiply_plain(intermediateResultsE[i],tmp,intermediateResultsE[i],env);
    //         intermediateResultsE[i].scale=intermediateResultsE[i].scale/RemoveScale;
    //     }
    //     cout<<"----scale "<<intermediateResultsE[0].scale<<endl;

    //     decrypt(intermediateResultsE,res1,env);
    //     cout<<"---------------------After " <<res1[0][0]<<endl;
    // }
    
    
    // replace to SRHE layer(Simulatiing ReLU by Homomorphic Encrytion)
    //simulate_relu(intermediateResultsE,env);
   
    {
        #pragma omp parallel for
        for (size_t i = 0; i < intermediateResultsE.size(); i++) {
            mod_switch_to_next_inplace(intermediateResultsE[i], env);
        }
    }

    


    // FC2 Layer
    {
        vector<SealBfvCiphertext> resultE;
        auto decor = make_decorator(fc, "FC2"); //matrix 10X845

        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;

        encode(params.FC2Weights, WeightsP, env, FC2_SCALE);
        encode(params.FC2Biases, BiasesP, env, intermediateResultsE[0].scale * FC2_SCALE);
        decor(getPointers(intermediateResultsE),
              WeightsP,
              BiasesP,
              intermediateResultsE.size(),
              resultE,
              env);

        intermediateResultsE.clear();
        intermediateResultsE = move(resultE);
         // Modulus switch
        {
            #pragma omp parallel for
            for (size_t i = 0; i < intermediateResultsE.size(); i++) {
                mod_switch_to_next_inplace(intermediateResultsE[i], env);
            }
        }
        cout<<"-Scale :"<<intermediateResultsE[0].scale<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }

    return intermediateResultsE;
}

int main()
{
    SealBfvEnvironment env = SealBfvEnvironment(
        POLY_MODULUS_DEGREE,
        PLAINTEXT_MODULUS);
    //ReverseScale=modInverse((size_t)147457*(size_t)163841,RemoveScale);
    // for(int i=0;i<PLAINTEXT_MODULUS.size();i++){
    //     cout<<"poly_modulus  "<<PLAINTEXT_MODULUS[i]<<endl;
    // }
    //cout<<"test   "<<modInverse(1001,2)<<endl;
    auto params = readParams("./resources/MnistSquare9665.txt");

    vector<vector<vector<double>>> data;
    vector<vector<size_t>> labels;
    readInputCryptonets(POLY_MODULUS_DEGREE, 1.0/255.0, data, labels,"./resources/MNIST-28x28-test.txt");


    // Batch processing

    size_t correct = 0;
    for (size_t batchIndex = 0; batchIndex < data.size(); batchIndex++)
    {
        // Encrypt(user)
        vector<SealBfvCiphertext> inputE;
        auto decorE = make_decorator(encrypt, "Encryption");
        decorE(data[batchIndex], inputE, env, INPUT_SCALE);
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(inputE[0].eVectors[0]) << endl;

        // Forward pass on the cloud(server)
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        auto intermediateResultsE = Cryptonets(params, inputE, env);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << setw(20) << "Total time: ";
        cout << elapsed_seconds.count() << " seconds" << std::endl;

        //output 845X1 ciphertext
        //Decrypt(user)
        vector<vector<double>> res;
        auto decorD=make_decorator(decrypt,"Decryption");
        decorD(intermediateResultsE, res, env);//res 10X8192 的double值

        auto predictions = hardmax(res); //1X8192

        for (size_t i = 0; i < predictions.size(); i++)
        {
            if (predictions[i] == labels[batchIndex][i])
            {
                correct++;
            }
        }
        //break;
    }
    cout << "Accuracy is " << correct << "/" << 10000;
    cout << "=" << (1.0 * correct)/10000 << endl;
    return 0;
}