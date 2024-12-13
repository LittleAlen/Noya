#include<time.h>
#include "NoyaNeuralNetworks.h"
#include "MnistDataPreprocess.h"
#include "SRHE.h"

using namespace std;
using namespace seal;
using namespace mycryptonets;

namespace {
    size_t POLY_MODULUS_DEGREE = 8192; 
    vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
        POLY_MODULUS_DEGREE, {18, 18});
    size_t BATCH_SIZE=1;
    double INPUT_SCALE = 16;
    double CONV_SCALE = 32*8;
    double FC1_SCALE = 32*32;
    double FC2_SCALE = 32*32;
    // vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
    // POLY_MODULUS_DEGREE, {18, 18});
    
    // double INPUT_SCALE = 1;
    // double CONV_SCALE =32.0;
    // double FC1_SCALE =1024.0;
    // double FC2_SCALE = 32.0;
}


vector<SealBfvCiphertext> Noya(const Params &params, const vector<SealBfvCiphertext> &inputE, const SealBfvEnvironment &env)
{
    vector<SealBfvCiphertext> intermediateResultsE;
    //vector<SealBfvCiphertext> CtestMemory;
    // vector<SealBfvPlaintext> PtestMemory;
    // for(int i=0;i<100;i++){
    //     PtestMemory.push_back(SealBfvPlaintext(66666,env,1));
    // }

    // std::stringstream ss;
    // inputE[0].eVectors[0].save(ss);
    // std::cout << "Size of ciphertext: " << ss.str().size()*inputE[0].eVectors.size() << std::endl;
    
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    // Conv Layer
    {
        vector<SealBfvCiphertext> resultE;
        auto decor = make_decorator(fc, "Conv");

        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;

        encode(params.convWeights, WeightsP, env, CONV_SCALE);
        encode_padding(params.convBiases, BiasesP, env, inputE[0].scale * CONV_SCALE,169,256);
        decor(getPointers(inputE),
              WeightsP,
              BiasesP,
              5 * 5,
              resultE,
              env);
        intermediateResultsE = move(resultE);

        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
   
    // replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
    simulate_relu(intermediateResultsE,env);
    // Modulus switch
    {
        #pragma omp parallel for
        for (size_t i = 0; i < intermediateResultsE.size(); i++) {
            mod_switch_to_next_inplace(intermediateResultsE[i], env);
        }
    }

    //Debug Only , we can use it for unlimited noisy budget
    // vector<vector<double>> res1;
    // decrypt(intermediateResultsE,res1,env);
    //encrypt(res1,intermediateResultsE,env,intermediateResultsE[0].scale);

    // FC1 Layer
    {
        start = std::chrono::steady_clock::now();

        vector<SealBfvCiphertext> resultE;

        vector<SealBfvPlaintext> WeightsP;
        SealBfvPlaintext BiasesP;
      
        for(int i=0;i<5*4;i++){
            int p1=i/5;
            int p2=i%5;
            vector<double> tmp(env.poly_modulus_degree, 0.0);
            for(int k=0;k<32;k++)
                for(int j=0;j<169;j++){
                    if(845*(p1+4*k)+p2*169+j<params.FC1Weights.size())
                        tmp[k*256+j]=params.FC1Weights[845*(p1+4*k)+p2*169+j];
                }
            WeightsP.emplace_back(move(SealBfvPlaintext(tmp,env,FC1_SCALE)));
        }
        
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        for (int i = 0; i < 32; i++) {
            vMsgs[i * 256] = 1.0;
        }
        SealBfvPlaintext slotMask(vMsgs,env,1.0);
       

        std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
      
        for(int i=0;i<100;i++){
            vMsgs[(i/4)*256+(i%4)]=params.FC1Biases[i];
        }
        
        BiasesP=SealBfvPlaintext(vMsgs,env,intermediateResultsE[0].scale * FC1_SCALE);
        
        vector<SealBfvCiphertext> & ciphertext= intermediateResultsE;
        SealBfvCiphertext tmp(WeightsP[0].scale);
       
        for(int i=0;i<4;i++){
           
            vector<SealBfvCiphertext> re;
            for(int j=0;j<5;j++){
                multiply_plain(ciphertext[j],WeightsP[i*5+j],tmp,env);
                re.emplace_back(move(tmp));
            }
            add_many(re,tmp,env);
            rotate_add(tmp,vector<int>({1,2,4,8,16,32,64,128}),tmp,env);
            multiply_plain(tmp,slotMask,tmp,env);
            int step=-i;
            rotate_inplace(tmp,step,env);
            resultE.emplace_back(move(tmp));
        }
        add_many(resultE,tmp,env);

        //collecting   because there is too much noise, we have to ignore it.
        // for(int i=1;i<32;i*=2){
        //     rotate_add(tmp,{i*(256-4)},tmp,env);
        // }
        // std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
        // for (int i = 0; i < 100; i++) {
        //     vMsgs[i] = 1.0;
        // }
        // slotMask= move(SealBfvPlaintext(vMsgs,env,1.0));
        // multiply_plain(tmp,slotMask,tmp,env);
        add_plain_inplace(tmp,BiasesP,env);

        rotate_add(tmp,{-4,-4*2,-4*4,-4*8},tmp,env);
        

        intermediateResultsE.clear();
        intermediateResultsE.emplace_back(move(tmp));
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;
        cout<<"FC1 compute time: "<<elapsed_seconds.count()<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }//100X2

    // Modulus switch
    {
        #pragma omp parallel for
        for (size_t i = 0; i < intermediateResultsE.size(); i++) {
            mod_switch_to_next_inplace(intermediateResultsE[i], env);
        }
    }
    // // Square activation layer
    // {
    //     auto decor = make_decorator(square_inplace_vec, "Square");

    //     decor(intermediateResultsE, env);

    //     cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    // }
    // replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
    simulate_relu(intermediateResultsE,env);
    
    // // Modulus switch
    // {
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < intermediateResultsE.size(); i++) {
    //         mod_switch_to_next_inplace(intermediateResultsE[i], env);
    //     }
    // }

    //Debug Only
    // vector<vector<double>> res6;
    // decrypt(intermediateResultsE,res6,env);
    // encrypt(res6,intermediateResultsE,env,intermediateResultsE[0].scale);


    // FC2 Layer
    {
        start = std::chrono::steady_clock::now();
        vector<SealBfvCiphertext> resultE;
        auto decor = make_decorator(fc, "FC2");
        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        
        //Stack representation reorganization   by duplicating
        //rotate_add(intermediateResultsE[0],{-128,-128*2,-128*4,-128*8},intermediateResultsE[0],env);

        
        std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
        // not collecting
        // for(int k=0;k<10;k++)
        //     for(int j=0;j<100;j++){
        //         vMsgs[k*128+j]=params.FC2Weights[k*100+j];
        //     }
         for(int k=0;k<10;k++)
            for(int j=0;j<100;j++){
                vMsgs[(j/4)*256+j%4+k*4]=params.FC2Weights[k*100+j];
            }
        WeightsP.emplace_back(move(SealBfvPlaintext(vMsgs,env,FC2_SCALE)));
        

        size_t Bias_Scale=intermediateResultsE[0].scale * FC2_SCALE;
        std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
       
        // for(int j=0;j<10;j++){
        //     vMsgs[j*128]=params.FC2Biases[j];
        // }
         for(int j=0;j<10;j++){
            vMsgs[j*4]=params.FC2Biases[j];
        }
        BiasesP.emplace_back(move(SealBfvPlaintext(vMsgs,env,Bias_Scale)));
        
        SealBfvCiphertext tmp(intermediateResultsE[0].scale);
        SealBfvCiphertext& ciphertext=intermediateResultsE[0];

        // std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
        //  for (int i =0; i <32; i++) {
        //     vMsgs[i * 256] = 1.0;
        // }
        // SealBfvPlaintext slotMask(vMsgs,env,1);

        for(int i=0;i<WeightsP.size();i++){
            multiply_plain(ciphertext,WeightsP[i],tmp,env);
            //rotate_add(tmp,vector<int>({1,2,4,8,16,32,64,128}),tmp,env);// comment it for optimization
            // multiply_plain(tmp,slotMask,tmp,env);  //for 1 ciphertext output
            add_plain_inplace(tmp,BiasesP[i],env);
            // if(i!=0)
            //     rotate_inplace(tmp,-i,env);
            resultE.emplace_back(move(tmp));
        }
        // this operation will run out of noise budget 
        // add_many(resultE,tmp,env); //for 1 ciphertext output
        //10 ... 10... 10...
        // intermediateResultsE.clear(); //for 1 ciphertext output
        // intermediateResultsE.emplace_back(move(tmp)); //for 1 ciphertext output
        intermediateResultsE=move(resultE);
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;

        cout<<"FC2 compute time: "<<elapsed_seconds.count()<<endl;
       
         // Modulus switch
        {
            #pragma omp parallel for
            for (size_t i = 0; i < intermediateResultsE.size(); i++) {
                mod_switch_to_next_inplace(intermediateResultsE[i], env);
            }
        }

        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }
    
    {
        start = std::chrono::steady_clock::now();
        srand((unsigned)time(NULL)); 
        double random=(1+(rand()%1000000))/1000000;
        SealBfvPlaintext random_value=SealBfvPlaintext(random,env,intermediateResultsE[0].scale);
        for(int i=0;i<intermediateResultsE.size();i++){
            add_plain_inplace(intermediateResultsE[i],random_value,env);
        }
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;
        cout<<"Random Layer compute time: "<<elapsed_seconds.count()<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
    // vector<vector<double>> res;
    // decrypt(intermediateResultsE, res, env);


    return intermediateResultsE;
}

int main()
{
    SealBfvEnvironment env = SealBfvEnvironment(
        POLY_MODULUS_DEGREE,
        PLAINTEXT_MODULUS);
    auto params = readParams("./resources/MnistReLUNoya.txt");

    vector<vector<vector<double>>> data;
    vector<vector<size_t>> labels;
    readInput(BATCH_SIZE, 1.0/255, data, labels,"./resources/MNIST-28x28-test.txt");

    // Batch processing

    size_t correct = 0;
    for (size_t batchIndex = 0; batchIndex < data.size(); batchIndex++)
    {
        //Conv Organization 
        vector<vector<vector<double>>>image_data(BATCH_SIZE,vector<vector<double>>(29, vector<double>(29, 0.0)));
        // zero-pad image

        for (int i = 1; i <= 28; i++) {
            for (int j = 1; j <= 28; j++) {
                for(int k=0;k<BATCH_SIZE;k++)
                    image_data[k][i][j] = data[batchIndex][k][(i - 1) * 28 + (j - 1)];
            }
        }
        vector<vector<double>> msgs = ConvertToConvolutionRepresentation(image_data);
       
         for(int i=1;i<32;i++){
            for(int j=0;j<256;j++){
                for(int k=0;k<msgs.size();k++){
                    msgs[k].push_back(msgs[k][j]);
                }
            }
        }
      
        assert(msgs[0].size()==8192);

        // Encrypt
        vector<SealBfvCiphertext> inputE;
        auto decor = make_decorator(encrypt, "Encryption");
        decor(msgs, inputE, env, INPUT_SCALE);
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(inputE[0].eVectors[0]) << endl;

        // Forward pass on the cloud
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        auto intermediateResultsE = Noya(params, inputE, env);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << setw(20) << "Total time: ";
        cout << elapsed_seconds.count() << " seconds" << std::endl;

        //output 10 ciphertext
        //Decrypt(user)
        vector<vector<double>> res;
        auto decorD=make_decorator(decrypt,"Decryption");
        decorD(intermediateResultsE, res, env);//res 1X8192

        // for(int i=0;i<32;i++){
        //     auto predicted = std::distance(res[0].begin()+i*256, std::max_element(res[0].begin()+i*256,res[0].begin()+i*256+10));
        //     cout << labels[batchIndex][i] << " vs " << predicted<< endl;
        //     correct += labels[batchIndex][i] == predicted ? 1 : 0;
        // }

        for(int i=0;i<BATCH_SIZE;i++){
            vector<double>val;
            for(int k=0;k<10;k++){
                 double tmp=0;
                for(int j=0;j<100;j++){
                    tmp+=res[0][(j/4)*256+j%4+k*4];
                }
                val.push_back(tmp);
            }
            auto predicted = std::distance(val.begin(), std::max_element(val.begin(),val.end()));
            cout << labels[batchIndex][i] << " vs " << predicted<< endl;
            correct += labels[batchIndex][i] == predicted ? 1 : 0;
        }
        cout << "Accuracy is " << correct << "/" << ((batchIndex+1)*BATCH_SIZE);
        cout << "=" << (1.0 * correct)/((batchIndex+1)*BATCH_SIZE) << endl;
        // break;
    }
     cout << "Accuracy is " << correct << "/" << 10000;
        cout << "=" << (1.0 * correct)/10000 << endl;
    return 0;
}