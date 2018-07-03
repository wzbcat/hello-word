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
void fun2Mat(MatrixXd &mat ,function<double(double)> & f)
 {
	 for(int i=0;i<mat.rows();i++)
	 {
		 for(int j=0;j<mat.cols();j++)
		 {
			mat(i,j)=f(mat(i,j));
		 }
	 }
 }
 void fun2Mat(VectorXd &vec ,function<double(double)> & f)
 {
	 for(int i=0;i<vec.size();i++)
	 {
		 vec(i)=f(vec(i));
	 }
 }
  void fun2Mat(VectorXd &vec ,function<double()> & f)
 {
	 for(int i=0;i<vec.size();i++)
	 {
		 vec(i)=f();
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
 
 
 
class DeepLearn
{
public:
	DeepLearn( initializer_list<int> nums)
	{
		active_funtion = [](double x)->double{ return 1/(1+exp(-x));};
		diff_active=[](double x)->double{ return x*(1-x); };
		lose_func = [&]()->double	{ return 0.5 *(Hide_Layers.back().layer - lable).squaredNorm();};
		deff_lose_func=[&]()->VectorXd{ return Hide_Layers.back().layer - lable ; };
		
		auto iter_num=nums.begin()+1;
		auto iter_num_pre=nums.begin();
		
		while( iter_num!=nums.end())
		{
			Hide_Layers.push_back (LayernNerve(*iter_num,*iter_num_pre));
			iter_num++;
			iter_num_pre++;
		}
	};
	~DeepLearn()
	{
		f_img.close();
		f_lable.close();
	}
	struct LayernNerve
	{
	public:
		LayernNerve(){}
		LayernNerve(int num,int pre_num)
		{
			layer=VectorXd::Zero(num);
			delta=VectorXd::Zero(num);			
			weight.resize(num,pre_num);
			bias.resize(num);
		}
		VectorXd layer;
		MatrixXd weight;
		VectorXd bias;
		VectorXd delta;
	};
	
	bool init_data(char * img_path , char * lable_path );
	
	void forward();
	void backward();
	void active_layer(VectorXd & vec);
	
	bool read_data();
	bool read_data(int i);
	
private:	
public:
	function<double(double)> active_funtion;
	function<double(double)> diff_active;
	function<VectorXd()> deff_lose_func;
	double eta;
	vector<LayernNerve> Hide_Layers;	
	ifstream  f_img;
	ifstream f_lable;
	


	function<double()> lose_func;
	VectorXd img;				//图片
	VectorXd lable;			//标签
};

bool  DeepLearn::init_data(char * img_path , char * lable_path )
{
	f_img.open(img_path , ios::binary | ios::in);
	f_lable.open(lable_path,ios::binary | ios::in);
	
	if(f_img.is_open() && f_lable.is_open())
	{
		f_img.seekg(16,ios::beg);
		f_lable.seekg(8,ios::beg);
		return true;
	}
	else
		return false;
}

void DeepLearn:: active_layer(VectorXd & vec)
{
	fun2Mat(vec,active_funtion);
}

void DeepLearn::forward()
{
	active_layer(img);
	Hide_Layers[0].layer= Hide_Layers[0].weight*img + Hide_Layers[0].bias;
	
	for(int i=1;i<Hide_Layers.size();i++)
	{
		Hide_Layers[i].layer=Hide_Layers[i].weight * Hide_Layers[i-1].layer +Hide_Layers[i].bias;
		active_layer(Hide_Layers[i].layer);
	}
}

void DeepLearn::backward()
{
	VectorXd lose=deff_lose_func();
	fun2Mat(Hide_Layers.back().layer,diff_active);
	
	Hide_Layers.back().delta=lose.array() * Hide_Layers.back().layer.array();
	
	for(int i=Hide_Layers.size()-1;i>1 ; i--)
	{
		fun2Mat(Hide_Layers[i].layer,diff_active);
		Hide_Layers[i].delta = (Hide_Layers[i+1].weight.transpose()*Hide_Layers[i+1].delta).array() * Hide_Layers[i].layer.array() ;
	}
}
bool DeepLearn::read_data()
{
	img=VectorXd::Zero(784);
	for(int i=0 ; i<img.size();i++)
	{
		img(i)=f_img.get();
	}
	
	lable=VectorXd::Zero(10);
	lable(f_lable.get())=1;
}

bool DeepLearn::read_data(int i)
{
	f_img.seekg(16 +i*784,ios::beg);
	f_lable.seekg(8+i,ios::beg);
	read_data();
}

int main(int argc, char **argv)
{/*
	DeepLearn dl({784,20,10});
	dl.init_data("D:\\Work\\Learn\\train_images","D:\\Work\\Learn\\train_labels");
	dl.read_data();
	show_img(dl.img);	


	dl.forward();
	dl.backward();
*/
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


