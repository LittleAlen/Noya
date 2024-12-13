#pragma once

#include "SealBfvCrtWrapper.h"

namespace mycryptonets
{
    //####################for operation in General
     void encrypt(const vector<vector<double>> &data,
                 vector<SealBfvCiphertext> &ciphertexts,
                 const SealBfvEnvironment &env,
                 double scale = 1.0)
    {
        size_t size = data.size();
        ciphertexts = vector<SealBfvCiphertext>(size, SealBfvCiphertext());

#pragma omp parallel for
        for (size_t i = 0; i < size; i++)
        {
            // cout<<"success "<<i<<" size:  "<<data[i].size()<<endl;
            ciphertexts[i] = SealBfvCiphertext(data[i], env, scale);
                                    
        }
    }

    void decrypt(const vector<SealBfvCiphertext> &ciphertexts,
                 vector<vector<double>> &data,
                 const SealBfvEnvironment &env)
    {
        size_t size = ciphertexts.size();
        assert(size > 0);
        data = vector<vector<double>>(size, {0});

#pragma omp parallel for
        for (size_t i = 0; i < size; i++)
        {
            if(ciphertexts[i].batchSize==0)
            {
                data[i]=vector<double>(env.poly_modulus_degree,0);
                continue;
            }
            data[i] = ciphertexts[i].decrypt(env);
        }
    }

    void encode(//按个数编码
        const vector<double> &data,
        vector<SealBfvPlaintext> &plaintexts,
        const SealBfvEnvironment &env,
        double scale = 1.0,int num=-1)//num parameter is used to control Paintext's valid value
    {
        size_t size = data.size();
        plaintexts = vector<SealBfvPlaintext>(size, SealBfvPlaintext());

#pragma omp parallel for
        for (size_t i = 0; i < size; i++)
        {   
            if(num==-1)
                plaintexts[i] = SealBfvPlaintext(data[i], env, scale);
            else{
                vector<double> tmp(num,data[i]);
                plaintexts[i] = SealBfvPlaintext(tmp, env, scale);
            }
        }
    }

    void batch_encode(
        const vector<vector<double>> &data,
        vector<SealBfvPlaintext> &plaintexts,
        const SealBfvEnvironment &env,
        double scale = 1.0)
    {
        size_t size = data.size();
        plaintexts = vector<SealBfvPlaintext>(size, SealBfvPlaintext());

#pragma omp parallel for
        for (size_t i = 0; i < size; i++)
        {
            plaintexts[i] = SealBfvPlaintext(data[i], env, scale);
        }
    }

    //####################for operation in Cryptonet

    void square_inplace(
        SealBfvCiphertext &ciphertext,
        const SealBfvEnvironment &env)
    {

        for (size_t i = 0; i < env.environments.size(); i++)
        {
            env.environments[i].evaluatorPtr->square_inplace(ciphertext.eVectors[i]);
            env.environments[i].evaluatorPtr->relinearize_inplace(ciphertext.eVectors[i], env.environments[i].relin_keys);
        }
        ciphertext.scale *= ciphertext.scale;
    }

    void mod_switch_to_next_inplace(
        SealBfvCiphertext &ciphertext,
        const SealBfvEnvironment &env)
    {
        if(ciphertext.batchSize==0)
            return;
        for (size_t i = 0; i < env.environments.size(); i++)
        {
            env.environments[i].evaluatorPtr->mod_switch_to_next_inplace(
                ciphertext.eVectors[i]);
        }
    }

    void multiply_plain(
        const SealBfvCiphertext &ciphertext,
        const SealBfvPlaintext &plaintext,
        SealBfvCiphertext &destination,
        const SealBfvEnvironment &env)
    {
        // cout<<"start multiply_plain   ";
        //         cout<<" "<<destination.batchSize<<" "<<destination.scale<<"  "<<destination.eVectors.size()<<endl;

        if(ciphertext.batchSize==0){
            destination.batchSize=0;
            destination.scale=ciphertext.scale*plaintext.scale;
            return;
            
        }
        size_t envCount = env.environments.size();
        vector<Ciphertext> eVectors(envCount, Ciphertext());

        for (size_t i = 0; i < envCount; i++)
        {
            env.environments[i].evaluatorPtr->multiply_plain(ciphertext.eVectors[i],
                                                             plaintext.pVectors[i],
                                                             eVectors[i]);
        }
        destination.eVectors = move(eVectors);
        destination.scale = ciphertext.scale * plaintext.scale;
        destination.batchSize = max(ciphertext.batchSize, plaintext.batchSize);
        // cout<<"end mutiply_plain"<<endl;
    }

    void add_many(const vector<SealBfvCiphertext> &ciphertexts,SealBfvCiphertext &destination,const SealBfvEnvironment &env)
    {
        // cout<<"start add_many"<<endl;

        assert(ciphertexts.size() > 0);
        //TODO assert all the scales are equal

        size_t envCount = env.environments.size();
        vector<Ciphertext> eVectors(envCount, Ciphertext());
        // cout<<"1"<<endl;
        int valid_index=0;
        while(valid_index<ciphertexts.size()&&ciphertexts[valid_index].batchSize==0)
            valid_index++;
        if(valid_index==ciphertexts.size()){
            destination.batchSize=0;
            destination.scale=ciphertexts[0].scale;
            // cout<<"Come  in add many"<<endl;
            return;
        }
        // cout<<" "<<destination.batchSize<<" "<<destination.scale<<"  "<<destination.eVectors.size()<<endl;

        for (size_t i = 0; i < envCount; i++)
        {
            eVectors[i] = ciphertexts[valid_index].eVectors[i];
            for (size_t j = valid_index+1; j < ciphertexts.size(); j++)
            {   
                if(ciphertexts[j].batchSize==0)
                    continue;
                env.environments[i].evaluatorPtr->add_inplace(eVectors[i], ciphertexts[j].eVectors[i]);
            }
        }

        // cout<<" "<<destination.batchSize<<" "<<destination.scale<<"  "<<destination.eVectors.size()<<endl;
           
        destination.eVectors = move(eVectors);
        destination.scale = ciphertexts[valid_index].scale;
        // batch size should be the largest
        vector<size_t> batchSizes;
        for_each(ciphertexts.begin(),
                 ciphertexts.end(),
                 [&batchSizes](const SealBfvCiphertext &ciphertext) { batchSizes.emplace_back(ciphertext.batchSize); });
        destination.batchSize = *max_element(batchSizes.begin(), batchSizes.end());

        // cout<<"end add_many"<<endl;
    }

    void add_plain_inplace(
        SealBfvCiphertext &ciphertext,
        const SealBfvPlaintext &plaintext,
        const SealBfvEnvironment &env)
    {
        // cout<<ciphertext.scale<<"  "<<plaintext.scale<<endl;
        assert(ciphertext.scale == plaintext.scale);
        if(ciphertext.batchSize==0)
        {
            // cout<<"error: E(0)+plaintext"<<endl;
            
            vector<double> res=plaintext.decode(env);
            res=vector<double>(plaintext.batchSize,res[0]);

            // //debug 
            // for(int i=0;i<res.size();i++){
            //     cout<<res[i]<<" ";
            // }
            // cout<<endl;
            ciphertext=SealBfvCiphertext(res,env,plaintext.scale);
            return;
        }
        

        for (size_t i = 0; i < env.environments.size(); i++)
        {
            env.environments[i].evaluatorPtr->add_plain_inplace(ciphertext.eVectors[i], plaintext.pVectors[i]);
        }
        ciphertext.batchSize = max(ciphertext.batchSize, plaintext.batchSize);
    }

    //####################for operation in Lola
    void square_inplace_vec(vector<SealBfvCiphertext> &ciphertexts,
                        const SealBfvEnvironment &env)
    {
#pragma omp parallel for
        for (size_t i = 0; i < ciphertexts.size(); i++)
        {
            square_inplace(ciphertexts[i], env);
        }
    }
    

   

   
    void rotate_inplace(SealBfvCiphertext &ciphertext,
                        int steps,
                        const SealBfvEnvironment &env)
    {
        for (size_t i = 0; i < env.environments.size(); i++)
        {
            env.environments[i].evaluatorPtr->rotate_rows_inplace(
                ciphertext.eVectors[i],
                steps,
                env.environments[i].gal_keys);
        }
      
    }

      void rotate_columns_inplace(SealBfvCiphertext &ciphertext,
                        const SealBfvEnvironment &env)
    {
        for (size_t i = 0; i < env.environments.size(); i++)
        {
            env.environments[i].evaluatorPtr->rotate_columns_inplace(
                ciphertext.eVectors[i],
                env.environments[i].gal_keys);
        }
      
    }
    void rotate_add(const SealBfvCiphertext &ciphertext,
                vector<int> steps,
                SealBfvCiphertext &destination,
                const SealBfvEnvironment &env)
    {
        destination = SealBfvCiphertext(ciphertext);
        for(auto step:steps){
            // cout<<"in rotate_add "<<step<<endl;
            SealBfvCiphertext tmp(destination);
            rotate_inplace(destination, step, env);
            vector<SealBfvCiphertext> all;
            all.emplace_back(move(tmp));
            all.emplace_back(move(destination));
            add_many(all,destination,env);
        }
    }

    //####################for operation in Noya
    void encode_padding(//encode value with padding
        const vector<double> &data,
        vector<SealBfvPlaintext> &plaintexts,
        const SealBfvEnvironment &env,
        double scale = 1.0,int num=169,int block=256)
    {
        size_t size = data.size();
        plaintexts = vector<SealBfvPlaintext>(size, SealBfvPlaintext());
        int rounds=env.poly_modulus_degree/block;
#pragma omp parallel for
        for (size_t i = 0; i < size; i++)
        {   
            vector<double> tmp(env.poly_modulus_degree,0);
            
            for(int j=0;j<rounds;j++)
                for(int p=0;p<num;p++)
                    tmp[j*block+p]=data[i];
            
            plaintexts[i] = SealBfvPlaintext(tmp, env, scale);
        }
    }
  

} // namespace mycryptonets