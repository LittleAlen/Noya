//the network we use is conv 25 conv 50 fc10
#include "NoyaNeuralNetworks.h"
#include "CifarDataPreprocess.h"
#include "SRHE.h"
using namespace std;
using namespace seal;
using namespace mycryptonets;

void encode(//按个数编码
        const vector<double> &data,
        vector<     long long     > &plaintexts,
        double scale = 1.0,int num=-1)//num parameter is used to control Paintext's valid value
    {
        size_t size = data.size();
        plaintexts = vector<     long long     >(size, 0);

#pragma omp parallel for
        for (size_t i = 0; i < size; i++)
        {   
            plaintexts[i] = data[i]*scale;
        }
    }
namespace {
    size_t POLY_MODULUS_DEGREE = 8192; 
    size_t BATCH_SIZE=32;
    // vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
    //     POLY_MODULUS_DEGREE, {18*2, 18*2});
    
    // double INPUT_SCALE = 1;
    // double CONV1_SCALE = 256;
    // double CONV2_SCALE = 512;
    // double FC_SCALE = 512;
    vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
    POLY_MODULUS_DEGREE, {18, 18});
    
    // double INPUT_SCALE = 1000;
    // double CONV1_SCALE = 100000;
    // double CONV2_SCALE = 100000;
    // double FC_SCALE = 100000;
     
    // double INPUT_SCALE = 1;
    // double CONV1_SCALE = 1;
    // double CONV2_SCALE = 1;
    // double FC_SCALE = 1;
    double INPUT_SCALE = 8*2;
    double CONV1_SCALE = 32*8;
    double CONV2_SCALE = 32*16;
    double FC_SCALE = 32*32;
    
   
}

double collect_value(vector<     long long     >& tmp,int k){
    double sum=0;
    for(int i=0;i<256;i++)
    {
        sum+=tmp[k*256+i];
    }
    
    return sum;
}
vector<vector<double>> Noya(const Params &params, const vector<vector<     long long     >> &inputE)
{
    vector<vector<     long long     >> intermediateResultsE;
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    double scale=0;
    // Conv1 Layer
    {
        vector<     long long     > WeightsP;
        vector<     long long     > BiasesP;
        start=std::chrono::steady_clock::now();
        scale=INPUT_SCALE * CONV1_SCALE;
        encode(params.conv1Weights, WeightsP,  CONV1_SCALE);
        encode(params.conv1Biases, BiasesP,  scale);
        

        for(int i=0;i<25;i++){
            vector<     long long     >tmp=vector<     long long     >(POLY_MODULUS_DEGREE,0);
            for(int j=0;j<3*4*4;j++){
                for(int k=0;k<POLY_MODULUS_DEGREE;k++){
                    tmp[k]+=inputE[j][k]*WeightsP[i*3*4*4+j];
                }
            }
            for(int k=0;k<POLY_MODULUS_DEGREE;k++){
                tmp[k]+=BiasesP[i];
            }
            intermediateResultsE.push_back(tmp);
        }
        end=std::chrono::steady_clock::now();
        elapsed_seconds=end-start;
        cout<<"Conv 1 time: "<<elapsed_seconds.count()<<endl;
       // cout<<"Conv1 finish"<<endl;
    }
    
    //debug
    // {
        
    //     stringstream s;
    //     for(int i=0;i<intermediateResultsE[0].size();i++)
    //         s<<(double)intermediateResultsE[0][i]/scale<<" ";
    //     s<<endl;
    //     ofstream outfile("./test_debug-conv1.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }
   
    // replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)

    {
        for(int i=0;i<intermediateResultsE.size();i++){
            for(int j=0;j<intermediateResultsE[0].size();j++){
                if(intermediateResultsE[i][j]<0)
                    intermediateResultsE[i][j]=0;
            }
        }
        cout<<"Relu finish"<<endl;
    }
    //debug
    // {
    //     stringstream s;
    //     for(int i=0;i<intermediateResultsE[0].size();i++)
    //         s<<(double)intermediateResultsE[0][i]/scale<<" ";
    //     s<<endl;
    //     ofstream outfile("./test_debug-srhe1.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }
  

   
    // Conv2 Layer
    {
        start = std::chrono::steady_clock::now();
        vector<vector<     long long     >> resultE;
        vector<vector<vector<     long long     >>> weight(50*7*7,vector<vector<     long long     >>(25,vector<     long long     >(POLY_MODULUS_DEGREE,0.0)));
        vector<vector<     long long     >> BiasesP(50*7*7,vector<     long long     >(POLY_MODULUS_DEGREE,0));
        scale=scale*CONV2_SCALE;
        #pragma omp parallel for
        for(int m=0;m<50;m++){
            #pragma omp parallel for
            for(int i=0;i<7;i++){
                for(int j=0;j<7;j++){
                    //compute the start point
                    int x=i*2;
                    int y=j*2;
                    for(int k=0;k<25;k++){
                        for(int ii=0;ii<4;ii++){
                            for(int jj=0;jj<4;jj++){
                                // if(x+ii<14&&y+jj<14) //we need to ignore some points, because the second convolution's padding is 4,which means 22x22
                                // {
                                    for(int b=0;b<BATCH_SIZE;b++){
                                        //cout<<"weight["<<m*7*7+i*7+j<<"]"<<"["<<k<<"]"<<"["<<b*256+(x+ii)*14+y+jj<<"]"<<"    params.conv2weight["<<m*83*10*10+k*10*10+ii*10+jj<<"]"<<endl;
                                         weight[m*7*7+i*7+j][k][b*256+(x+ii)*16+y+jj]=CONV2_SCALE*params.conv2Weights[m*25*4*4+k*4*4+ii*4+jj];}
                                // }
                            }
                        }
                    }
                }
            }
        }

        #pragma omp parallel for
        for(int i=0;i<50;i++){
            for(int j=0;j<7*7;j++){
                for(int k=0;k<BATCH_SIZE;k++){
                    BiasesP[i*7*7+j][k*256]=params.conv2Biases[i]*scale; 
                }
            }
        }

        for(int i=0;i<50*7*7;i++){
            vector<     long long     >tmp(POLY_MODULUS_DEGREE,0);
            for(int j=0;j<25;j++){
                for(int k=0;k<POLY_MODULUS_DEGREE;k++){
                    tmp[k]+=intermediateResultsE[j][k]*weight[i][j][k];
                }
            }
            for(int k=0;k<POLY_MODULUS_DEGREE;k++){
                tmp[k]+=BiasesP[i][k];
            }

            // if(i==0)
            // {
            //     stringstream s;
            //     for(int i=0;i<tmp.size();i++)
            //         s<<(double)tmp[i]/scale<<" ";
            //     s<<endl;
            //     ofstream outfile("./test_tmp.txt",ios::out);
            //     outfile << s.str();
            //     outfile.close();
            // }

            resultE.emplace_back(move(tmp));
        }
   
        intermediateResultsE.clear();
        vector<vector<     long long     >>().swap(intermediateResultsE);
        
        
       
        //convert it to 10 ciphertext
        for(int i=0;i<10;i++){
            vector<     long long     >tmp(POLY_MODULUS_DEGREE,0);
            for(int j=0;j<256;j++){
                if(i*256+j<50*7*7){
                    for(int k=0;k<BATCH_SIZE;k++){
                        tmp[k*256+j]=collect_value(resultE[i*256+j],k);
                    }
                }
            }
            intermediateResultsE.emplace_back(move(tmp));
        }
        

        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;
        cout<<"Conv2 compute time: "<<elapsed_seconds.count()<<endl;
        
    }

    //debug
    // {
    //    stringstream s;
    //     for(int i=0;i<intermediateResultsE[0].size();i++)
    //         s<<(double)intermediateResultsE[0][i]/scale<<" ";
    //     s<<endl;
    //     s<<endl;
    //     ofstream outfile("./test_debug-conv2.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }

  
    
    //replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
    {
        for(int i=0;i<intermediateResultsE.size();i++){
            for(int j=0;j<intermediateResultsE[0].size();j++){
                if(intermediateResultsE[i][j]<0)
                    intermediateResultsE[i][j]=0;
            }
        }
    }
    
    //debug
    // {
    //     stringstream s;
    //     for(int i=0;i<intermediateResultsE[0].size();i++)
    //         s<<(double)intermediateResultsE[0][i]/scale<<" ";
    //     s<<endl;
       
    //     ofstream outfile("./test_debug-srhe2.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }
  

    // FC Layer
    {
        start = std::chrono::steady_clock::now();
        vector<vector<     long long     >> resultE;
        vector<vector<     long long     >> WeightsP;
        vector<vector<     long long     >> BiasesP(10,vector<     long long     >(POLY_MODULUS_DEGREE,0));
        vector<vector<     long long     >>weight(100,vector<     long long     >(POLY_MODULUS_DEGREE,0));
        scale=scale*FC_SCALE;
        // encode(params.FCWeights,WeightsP,env,FC_SCALE);
        #pragma omp parallel for
        for(int n=0;n<10;n++){
            for(int i=0;i<10;i++){
                for(int j=0;j<256;j++){
                    if(i*256+j<50*7*7){
                        for(int k=0;k<BATCH_SIZE;k++){
                            weight[n*10+i][k*256+j]=params.FCWeights[n*50*7*7+i*256+j]*FC_SCALE;
                        }
                    }else{break;}
                }
            }
        }
        for(int i=0;i<10;i++){
            for(int j=0;j<BATCH_SIZE;j++)
                BiasesP[i][j*256]=params.FCBiases[i]*scale;
        }

        for(int i=0;i<10;i++){
            vector<     long long     > tmp(POLY_MODULUS_DEGREE,0);
            for(int j=0;j<10;j++){
                for(int k=0;k<POLY_MODULUS_DEGREE;k++){
                    tmp[k]+=intermediateResultsE[j][k]*weight[i*10+j][k];
                }
            }
            for(int k=0;k<POLY_MODULUS_DEGREE;k++)
                tmp[k]+=BiasesP[i][k];
            resultE.emplace_back(move(tmp));
        }

        intermediateResultsE.clear();
        vector<vector<     long long     >>().swap(intermediateResultsE);
        intermediateResultsE=move(resultE);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        cout<<"FC compute time: "<<elapsed_seconds.count()<<endl;
                
    }
    //debug
    // {
    //     stringstream s;
    //     for(int i=0;i<intermediateResultsE[0].size();i++)
    //         s<<(double)intermediateResultsE[0][i]/scale<<" ";
    //     s<<endl;
    //     ofstream outfile("./test_debug-fc.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    //}
    
    // vector<vector<double>> res;
    // decrypt(intermediateResultsE, res, env);
    vector<vector<double>> ans(10,vector<double>(POLY_MODULUS_DEGREE,0));

    for(int i=0;i<10;i++)
        for(int j=0;j<POLY_MODULUS_DEGREE;j++)
            ans[i][j]=intermediateResultsE[i][j]/scale;

    return ans;
}

int main()
{
    SealBfvEnvironment env = SealBfvEnvironment(
        POLY_MODULUS_DEGREE,
        PLAINTEXT_MODULUS);

    auto params = readParams("./resources/CifarNoyaTest.txt");
    //auto params = readParams("./resources/model_params7229_c25_c50_f10.txt");

    vector<vector<vector<double>>> data;
    vector<vector<size_t>> labels;
    readInput(BATCH_SIZE, 1.0/255, data, labels,"./resources/CIFAR-3x32x32-test.txt");
    // Batch processing

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
                    for(int k=0;k<BATCH_SIZE;k++){
                        // image_data[k][d][i][j] = data[batchIndex][k][d*32*32+(i - 1) * 32 + (j - 1)];
                        double t=data[batchIndex][k][d*32*32+(i - 1) * 32 + (j - 1)];
                        int tmp=t*16;
                        image_data[k][d][i][j]=(double)tmp/16.0;
                    }
                        
                }
            }
        }
        vector<vector<double>> msgs =  ConvertToConvolutionRepresentationNew(image_data);
                    // for(int i=0;i<256*3;i++)
                    //     cout<<msgs[0][i]<<" ";
                    // cout<<endl;
        assert(msgs[0].size()==POLY_MODULUS_DEGREE);
        cout<<"finish ConvertToConvolutionRepresentation"<<endl;
        // Encrypt
        vector<vector<     long long     >> inputE(3*4*4,vector<     long long     >(POLY_MODULUS_DEGREE,0));
        for(int i=0;i<3*4*4;i++)
            for(int j=0;j<POLY_MODULUS_DEGREE;j++)
                inputE[i][j]=msgs[i][j]*INPUT_SCALE;
        // Forward pass on the cloud
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        auto intermediateResultsE = Noya(params, inputE);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << setw(20) << "Total time: ";
        cout << elapsed_seconds.count() << " seconds" << std::endl;

        //output 10 ciphertext
        //Decrypt(user)
        vector<vector<double>> res=intermediateResultsE;
      
        // for(int i=0;i<256*2;i++)
        //     cout<<res[0][i]<<" ";
        //     cout<<endl;
        // cout<<res.size()<<"  "<<res[0].size()<<endl;
        for(int i=0;i<BATCH_SIZE;i++){
            vector<double>val;
            for(int j=0;j<10;j++){
                double node=0;
                for(int k=0;k<256;k++)
                    node+=res[j][i*256+k];
                val.push_back(node);
                // cout<<node<<" ";
            }
            // cout<<endl;
            auto predicted = std::distance(val.begin(), std::max_element(val.begin(),val.end()));
            cout << labels[batchIndex][i] << " vs " << predicted<< endl;
            correct += labels[batchIndex][i] == predicted ? 1 : 0;
        }
         cout << "Accuracy is " << correct << "/" << (batchIndex+1)*BATCH_SIZE;
        cout << "=" << (1.0 * correct)/((batchIndex+1)*BATCH_SIZE) << endl;
        // break;
    }
     cout << "Accuracy is " << correct << "/" << 10000;
        cout << "=" << (1.0 * correct)/10000 << endl;
    return 0;
}