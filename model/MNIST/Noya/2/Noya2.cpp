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
    size_t BATCH_SIZE=16;
    int BLOCKS=2;
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
        vector<vector<double>> weight(50*5,vector<double>(8192,0));
        vector<SealBfvPlaintext> WeightsP;
        SealBfvPlaintext BiasesP;
        
        for(int i=0;i<100;i++){
            int v1=i/50;
            int v2=i%50;
            for(int j=0;j<BATCH_SIZE;j++) {  
                for(int p=0;p<169;p++)
                    for(int q=0;q<5;q++)
                        weight[v2*5+q][j*256*BLOCKS+256*v1+p]=params.FC1Weights[845*i+q*169+p];
            }
        }
        for(int i=0;i<50*5;i++)
            WeightsP.emplace_back(move(SealBfvPlaintext(weight[i],env,FC1_SCALE)));
        weight.clear();
        vector<vector<double>>().swap(weight);
        
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        for (int i = 0; i < 32; i++) {
            vMsgs[i * 256] = 1.0;
        }
        SealBfvPlaintext slotMask(vMsgs,env,1.0);
       

        std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
      
        for(int b=0;b<BATCH_SIZE;b++){
            for(int i=0;i<100;i++){
                vMsgs[b*256*BLOCKS+(i/50)*256+(i%50)]=params.FC1Biases[i];
            }
        }

        BiasesP=SealBfvPlaintext(vMsgs,env,intermediateResultsE[0].scale * FC1_SCALE);
        
        vector<SealBfvCiphertext> & ciphertext= intermediateResultsE;
        SealBfvCiphertext tmp(WeightsP[0].scale);
       
        for(int i=0;i<50;i++){
           
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
        SealBfvCiphertext tmp1=tmp;
        rotate_inplace(tmp1,-200,env);
        rotate_add(tmp,{-50,-50*2},tmp,env);
        resultE.clear();
        resultE.emplace_back(move(tmp1));
        resultE.emplace_back(move(tmp));
        add_many(resultE,tmp,env);

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
        vector<vector<double>>weight(2,vector<double>(8192,0));
        vector<SealBfvPlaintext> BiasesP;
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        
        //Stack representation reorganization   by duplicating
        //rotate_add(intermediateResultsE[0],{-128,-128*2,-128*4,-128*8},intermediateResultsE[0],env);

        
        // std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
        // not collecting
        // for(int k=0;k<10;k++)
        //     for(int j=0;j<100;j++){
        //         vMsgs[k*128+j]=params.FC2Weights[k*100+j];
        //     }
        for(int b=0;b<BATCH_SIZE;b++){
            for(int k=0;k<10;k++)
            for(int j=0;j<100;j++){
                weight[k/5][b*256*BLOCKS+(j/50)*256+(k%5)*50+j%50]=params.FC2Weights[k*100+j];
            }
        }
        for(int i=0;i<2;i++)
            WeightsP.emplace_back(move(SealBfvPlaintext(weight[i],env,FC2_SCALE)));
        

        size_t Bias_Scale=intermediateResultsE[0].scale * FC2_SCALE;
        std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
       
        // for(int j=0;j<10;j++){
        //     vMsgs[j*128]=params.FC2Biases[j];
        // }
       for(int k=0;k<2;k++){
        for(int b=0;b<BATCH_SIZE;b++){
            for(int j=0;j<5;j++){
            vMsgs[b*256*BLOCKS+j*50]=params.FC2Biases[k*5+j];
        }
        }
        BiasesP.emplace_back(move(SealBfvPlaintext(vMsgs,env,Bias_Scale)));
       }
        
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
        {
            vector<vector<double>> tmp_msgs(msgs.size(),vector<double>(8192,0));
            int step=(POLY_MODULUS_DEGREE/256)/BATCH_SIZE;
            for(int i=0;i<BATCH_SIZE;i++){
                for(int j=0;j<256;j++){
                    for(int k=0;k<msgs.size();k++){
                        for(int p=0;p<step;p++){
                            tmp_msgs[k][i*(256*step)+256*p+j]=msgs[k][i*256+j];
                        }
                    }
                }
            }
            msgs=tmp_msgs;
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
                    tmp+=res[k/5][i*256*BLOCKS+(j/50)*256+(k%5)*50+j%50];
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