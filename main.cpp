#include "eigen/dense"
#include<iomanip>
#include <fstream>
#include <iostream>
#include <random>
#include <ctime>
#include <initializer_list>
#include <vector>
#include <cassert>
#include <functional>
using namespace Eigen;
using namespace std;
void show_img(VectorXd & img)
{
	int k=0;
	for(int i=0 ; i<28 ; i++)
	{
		for(int j=0 ; j<28 ; j++)
		{
			if(img(k++))
				cout <<1;
			else
				cout <<" ";
		}
		cout <<endl;
	}
}


struct data
{
	int lablel;
	VectorXd image;
};
 
 class NetWork
 {
public:
	template< typename ... T>
	 NetWork(  T ... nums)
	 {
		srand(time(0));
		num_layers =sizeof...(nums);
		sigmoid = [](double x)->double{ return 1/(1+exp(-x));};
		sigmoid_prime=[](double x)->double{ return x*(1-x); };
		init(nums...);
	 }
	 
	 template<typename T ,typename ...Type>
	 void init(T first ,T next , Type ...rest)
	 {
		 bias.push_back(VectorXd(next));
		 weights.push_back(MatrixXd(next , first));
		 init_normal(first , next);
		 init(next ,rest...);
	 }
	template<typename T>
	void init(T first ,T next)
	{
		bias.push_back(VectorXd(next));
		weights.push_back(MatrixXd(next , first));
		init_normal(first , next);
	}
	 
	 void init_normal(int first ,int next)
	 {
		for(int i=0 ; i<next; i++)
		{
			bias.back()(i)=normal_rand(engine);
			for (int j=0 ; j<first ; j++)
			{
				weights.back()(i,j)=normal_rand(engine);
			}
		}
	}
	 
	 void feedforward(VectorXd & active)
	 {
		 for(int i=0 ; i<num_layers-1;i++)
		 {
			 VectorXd tmp=weights[i]*active;
			 tmp =tmp +bias[i];
			 fun2Mat(tmp  ,sigmoid);
		 }
	 }
	 
	 void SOD(vector<data> & training_data , int epochs ,int mini_batch_size ,double eta)
	 {
		 
	 }
	 
	 void update_mini_batch(vector<data> & mini_batch, double eta)
	 {
		 
	 }
	 
	void backprop(double x ,double y)
	{
		
	}
	
	int evaluate()
	{
		return 0;
	}
	 
public:
	int num_layers;
	default_random_engine engine;
	normal_distribution<> normal_rand;
	function<double(double)> sigmoid;
	function<double(double)> sigmoid_prime;
	 vector<MatrixXd> weights;
	 vector<VectorXd> bias;
 };
 
 

int main(int argc, char **argv)
{
	ifstream f_lables("D:\\Work\\Learn\\train_labels",ios::binary | ios::in);
	ifstream f_images("D:\\Work\\Learn\\train_images",ios::binary | ios::in);
	
	f_images.seekg(16,ios::beg);
	f_lables.seekg(8,ios::beg);
	
	vector<data> training_data;
	
	VectorXd img_tmp=VectorXd::Zero(784);
	int lable_tmp=0;
	data tmp;
	for(int j=0 ; j<1000 ; j++)
	{
		for(int i=0 ; i<784;i++)
		{
			img_tmp(i)=f_images.get();
		}
		lable_tmp=f_lables.get();	
		tmp.image=img_tmp;
		tmp.lablel=lable_tmp;
		training_data.push_back(tmp);
	}
	
	NetWork net(784,30,10);
	cout.setf(ios::fixed);

	
	cout  <<net.bias[0] <<endl <<endl;
	cout <<net.bias[1] <<endl <<endl;
	cout <<net.weights[0] <<endl <<endl;
	cout <<net.weights[1 ] <<endl <<endl;

	return 0;
}


