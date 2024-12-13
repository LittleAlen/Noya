#include "LoLaNeuralNetworks.h"
#include "CifarDataPreprocess.h"
#include "SRHE.h"
using namespace std;
using namespace seal;
using namespace mycryptonets;

namespace {
    size_t POLY_MODULUS_DEGREE = 16384; 
    size_t BATCH_SIZE=1;
    vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
        POLY_MODULUS_DEGREE, {32,32,32});
    
    double INPUT_SCALE = 8;
    double CONV1_SCALE = 256;
    double CONV2_SCALE = 512;
    double FC_SCALE = 512;
    
    // vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
    // POLY_MODULUS_DEGREE, {18, 18});
    
    // double INPUT_SCALE = 1;
    // double CONV_SCALE =32.0;
    // double FC1_SCALE =1024.0;
    // double FC2_SCALE = 32.0;
}


vector<SealBfvCiphertext> LoLa(const Params &params, const vector<SealBfvCiphertext> &inputE, const SealBfvEnvironment &env)
{
    cout<<"start LoLa"<<endl;
    vector<SealBfvCiphertext> intermediateResultsE;

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    // Conv1 Layer
    {
        vector<SealBfvCiphertext> resultE;
        auto decor = make_decorator(fc, "Conv1");

        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;

        encode(params.conv1Weights, WeightsP, env, CONV1_SCALE);
        encode_padding(params.conv1Biases, BiasesP, env, inputE[0].scale * CONV1_SCALE,196,env.poly_modulus_degree);
        decor(getPointers(inputE),
              WeightsP,
              BiasesP,
              3*8*8,
              resultE,
              env);

        SealBfvCiphertext tmp(resultE[0].scale);
        for(int i=1;i<83;i++){
            if(i==41){
                rotate_inplace(resultE[i],-196*i,env);
                vector<double>vMsgs1(env.poly_modulus_degree,0.0);
                vector<double>vMsgs2(env.poly_modulus_degree,0.0);
                for(int k=0;k<40;k++)
                    vMsgs1[i]=1;
                for(int k=0;k<156;k++)
                    vMsgs2[196*i+k]=1;
                SealBfvPlaintext slot1(vMsgs1,env,1);
                SealBfvPlaintext slot2(vMsgs2,env,2);
                SealBfvCiphertext tmp_low(resultE[i].scale);
                SealBfvCiphertext tmp_up(resultE[i].scale);
                multiply_plain(resultE[i],slot1,tmp_low,env);
                multiply_plain(resultE[i],slot2,tmp_up,env);
                rotate_columns_inplace(tmp_low,env);
                vector<SealBfvCiphertext> val;
                val.emplace_back(move(tmp_low));
                val.emplace_back(move(tmp_up));
                add_many(val,resultE[i],env);
            }
            else if(i<41)
                    rotate_inplace(resultE[i],-196*i,env);
            else{
                    rotate_columns_inplace(resultE[i],env);
                    rotate_inplace(resultE[i],-40-196*(i-42),env);
                }
            }
        add_many(resultE,tmp,env);
        intermediateResultsE.clear();
        vector<SealBfvCiphertext>().swap(intermediateResultsE);
        intermediateResultsE.emplace_back(move(tmp));
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
    

    // Square activation layer
    // {
    //     auto decor = make_decorator(square_inplace_vec, "Square");

    //     decor(intermediateResultsE, env);
    //     cout<<"-Scale :"<<intermediateResultsE[0].scale<<endl;
    //     cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    // }
    
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
    // encrypt(res1,intermediateResultsE,env,intermediateResultsE[0].scale);
    // cout<<"start conv2"<<endl;
    // Conv2 Layer
    {
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

        vector<SealBfvCiphertext> resultE;

        vector<vector<double>> weight(112*7*7,vector<double>(env.poly_modulus_degree,0.0));
        vector<SealBfvPlaintext>WeightsP(112*7*7,SealBfvPlaintext());
        vector<SealBfvPlaintext> BiasesP(112*7*7,SealBfvPlaintext());
        cout<<"parameters preprocess"<<endl;
        #pragma omp parallel for
        for(int m=0;m<112;m++){
            #pragma omp parallel for
            for(int i=0;i<7;i++){
                for(int j=0;j<7;j++){
                    //compute the start point
                    int x=i*2;
                    int y=j*2;
                    
                    for(int k=0;k<83;k++){
                        for(int ii=0;ii<10;ii++){
                            for(int jj=0;jj<10;jj++){
                                if(x+ii<14&&y+jj<14) //we need to ignore some points, because the second convolution's padding is 4,which means 22x22
                                {
                                    //cout<<"weight["<<m*7*7+i*7+j<<"]"<<"["<<k<<"]"<<"["<<b*256+(x+ii)*14+y+jj<<"]"<<"    params.conv2weight["<<m*83*10*10+k*10*10+ii*10+jj<<"]"<<endl;
                                    weight[m*7*7+i*7+j][k*196+(x+ii)*14+y+jj]=params.conv2Weights[m*83*10*10+k*10*10+ii*10+jj];
                                } 
                            }
                        }
                    }
                }
            }
        }
        #pragma omp parallel for
        for(int i=0;i<112*7*7;i++){ 
            WeightsP[i]=SealBfvPlaintext(weight[i],env,CONV2_SCALE);
        }
        weight.clear();
        vector<vector<double>>().swap(weight);

        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        #pragma omp parallel for
        for(int i=0;i<112;i++){
            #pragma omp parallel for
            for(int j=0;j<7*7;j++){
                vMsgs[0]=params.conv2Biases[i]; 
                BiasesP[i*7*7+j]=SealBfvPlaintext(vMsgs,env,intermediateResultsE[0].scale*CONV2_SCALE);
            }
        }
        cout<<"finished, start fc"<<endl;
        fc(getPointers(intermediateResultsE),WeightsP,BiasesP,1,resultE,env);
        intermediateResultsE.clear();
        vector<SealBfvCiphertext>().swap(intermediateResultsE);
        //convert to 1 ciphertext
        vector<double>vMsg1(env.poly_modulus_degree,0);
        vMsg1[0]=1;
        vMsg1[8192]=1;
        SealBfvPlaintext slotMask(vMsg1,env,1);
        SealBfvCiphertext tmp;
        vector<int>steps={1,2,4,8,16,32,64,128,256,512,1024,2048,4096};
        for(int i=0;i<resultE.size();i++){
            vector<SealBfvCiphertext> all;
            rotate_add(resultE[i],steps,tmp,env);
            multiply_plain(tmp,slotMask,tmp,env);
            SealBfvCiphertext tmp1=tmp;
            rotate_columns_inplace(tmp,env);
            all.emplace_back(move(tmp1));
            all.emplace_back(move(tmp));
            add_many(all,tmp,env);
            rotate_inplace(tmp,-1*i,env);
            intermediateResultsE.emplace_back(move(tmp));
        } 
        resultE.clear();
        vector<SealBfvCiphertext>().swap(intermediateResultsE);
        add_many(intermediateResultsE,tmp,env);
        intermediateResultsE.clear();
        vector<SealBfvCiphertext>().swap(intermediateResultsE);
        intermediateResultsE.emplace_back(move(tmp));
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout<<"Conv2 compute time: "<<elapsed_seconds.count()<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }

    // Modulus switch
    {
        #pragma omp parallel for
        for (size_t i = 0; i < intermediateResultsE.size(); i++) {
            mod_switch_to_next_inplace(intermediateResultsE[i], env);
        }
    }
    //Debug Only
    // vector<vector<double>> res6;
    // decrypt(intermediateResultsE,res6,env);
    // for(int i=0;i<res6.size();i++)
    //     for(int j=0;j<res6[i].size();i++)
    //         res6[i][j]=res6[i][j]*res6[i][j];
    // encrypt(res6,intermediateResultsE,env,intermediateResultsE[0].scale);

    // Square activation layer
    // {
    //     auto decor = make_decorator(square_inplace_vec, "Square");

    //     decor(intermediateResultsE, env);

    //     cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    // }
    //replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
    simulate_relu(intermediateResultsE,env);
    
    // // Modulus switch
    // {
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < intermediateResultsE.size(); i++) {
    //         mod_switch_to_next_inplace(intermediateResultsE[i], env);
    //     }
    // }

    

    // FC Layer
    {
        cout<<" come to FC layer"<<endl;
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        vector<SealBfvCiphertext> resultE;
        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;
        vector<vector<double>> weight(10,vector<double>(env.poly_modulus_degree,0));
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        //encode(params.FCWeights,WeightsP,env,FC_SCALE);
        for(int i=0;i<10;i++){
            for(int j=0;j<112*7*7;j++){
                weight[i][j]=params.FCWeights[i*10+j];
            }
        }
        weight.clear();
        vector<vector<double>>().swap(weight);

        for(int i=0;i<10;i++){
            vMsgs[0]=params.FCBiases[i];
            BiasesP.emplace_back(move(SealBfvPlaintext(vMsgs,env,intermediateResultsE[0].scale*FC_SCALE)));
        }
        fc(getPointers(intermediateResultsE),WeightsP,BiasesP,1,resultE,env);
        intermediateResultsE.clear();
        vector<SealBfvCiphertext>().swap(intermediateResultsE);
        intermediateResultsE=move(resultE);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        cout<<"FC compute time: "<<elapsed_seconds.count()<<endl;
       
         // Modulus switch
        {
            #pragma omp parallel for
            for (size_t i = 0; i < intermediateResultsE.size(); i++) {
                mod_switch_to_next_inplace(intermediateResultsE[i], env);
            }
        }

        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }
    
    // vector<vector<double>> res;
    // decrypt(intermediateResultsE, res, env);

    return intermediateResultsE;
}

int Test(const SealBfvEnvironment &env)
{
    //It is used to test how many ciphertext memory can maintain in our computer
    vector<double>p(16384,6);
    SealBfvCiphertext tmp(p,env,100000);
    vector<SealBfvCiphertext>all;
    while(true){
        all.push_back(tmp);
        cout<<"size : "<<all.size()<<endl;
    }
    return 0;
}
int main()
{
    
    SealBfvEnvironment env = SealBfvEnvironment(
        POLY_MODULUS_DEGREE,
        PLAINTEXT_MODULUS);

    auto params = readParams("./resources/CifarSquareLoLa.txt");
    vector<vector<vector<double>>> data;
    vector<vector<size_t>> labels;
    readInput(BATCH_SIZE, 1.0/256, data, labels,"./resources/CIFAR-3x32x32-test.txt");
    // Batch processing

    //Test(env);
    size_t correct = 0;
    for (size_t batchIndex = 0; batchIndex < data.size(); batchIndex++)
    {   
        cout<<"start a batch "<<batchIndex<<endl;
        //Conv Organization 
        vector<vector<vector<vector<double>>>>image_data(BATCH_SIZE,vector<vector<vector<double>>>(3, vector<vector<double>>(34, vector<double>(34,0.0))));
        // zero-pad image
        for(int d=0;d<3;d++){
            for (int i = 1; i <= 32; i++) {
                for (int j = 1; j <= 32; j++) {
                    for(int k=0;k<BATCH_SIZE;k++)
                        image_data[k][d][i][j] = data[batchIndex][k][d*1024+(i - 1) * 32 + (j - 1)];
                }
            }
        }
        vector<vector<double>> msgs = ConvertToConvolutionRepresentation(image_data);
        cout<<"finish ConvertToConvolutionRepresentation"<<endl;
        // Encrypt
        vector<SealBfvCiphertext> inputE;
        auto decor = make_decorator(encrypt, "Encryption");
        decor(msgs, inputE, env, INPUT_SCALE);
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(inputE[0].eVectors[0]) << endl;

        // Forward pass on the cloud
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        auto intermediateResultsE = LoLa(params, inputE, env);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << setw(20) << "Total time: ";
        cout << elapsed_seconds.count() << " seconds" << std::endl;

        //output 10 ciphertext
        //Decrypt(user)
        vector<vector<double>> res;
        auto decorD=make_decorator(decrypt,"Decryption");
        decorD(intermediateResultsE, res, env);


        for(int i=0;i<BATCH_SIZE;i++){
            vector<double>val;
            for(int j=0;j<10;j++){
                double node=0;
                for(int k=0;k<112*7*7;k++)
                    node+=res[j][k];
                val.push_back(node);
                cout<<node<<" ";
            }
            cout<<endl;
            auto predicted = std::distance(val.begin(), std::max_element(val.begin(),val.end()));
            cout << labels[batchIndex][i] << " vs " << predicted<< endl;
            correct += labels[batchIndex][i] == predicted ? 1 : 0;
        }
       
        // break;
    }
     cout << "Accuracy is " << correct << "/" << 10000;
        cout << "=" << (1.0 * correct)/10000 << endl;
    return 0;
}