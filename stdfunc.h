/*This class defines commenly used functions*/
#pragma once
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cerrno>

namespace stdfunc
{
	/*PI*/
	static const double LOW = 0.02425;
	static const double HIGH = 0.97575;
	static const double PI = atan(1.0)*4;

	template <typename T>
	inline double variance(const T *array_start,const  T *array_end)
	{
		double sum(0);
		int count(0);
		for(const T *p=array_start;p!=array_end;p++)
		{
			sum+=*p;
			count++;
		}
		double mean=sum/count;
		double sum_diff(0);
		for(const T *p=array_start;p!=array_end;p++)
		{
			sum_diff+=(*p-mean)*(*p-mean);
		}
		return sum_diff/(count-1);
	}
	
	/*given its mean value*/
	template <typename T>
	inline double variance(const T *array_start,const T *array_end, double mean)
	{
		double sum_diff(0);
		int count(0);
		for(const T *p=array_start;p!=array_end;p++)
		{
			sum_diff+=(*p-mean)*(*p-mean);
			count++;
		}
		return sum_diff/(count-1);
	}

	/*Given a value, mean and std, total # of measures, check whether the value is a outlier using Chauvenet's criterion*/
	template<typename T>
	inline bool checkOutlier_Chauvenet(T value, double mean, double std, int num)
	{
		static const double threshold = 0.5;
		static const double gaussian_table[] =
		{//0.00 0.01	0.02	0.03	0.04	0.05	0.06	0.07	0.08	0.09
		0.5,	0.504,	0.508,	0.512,	0.516,	0.5199,	0.5239,	0.5279,	0.5319,	0.5359,//0.0
		0.5398,	0.5438,	0.5478,	0.5517,	0.5557,	0.5596,	0.5636,	0.5675,	0.5714,	0.5753,//0.1
		0.5793,	0.5832,	0.5871,	0.591,	0.5948,	0.5987,	0.6026,	0.6064,	0.6103,	0.6141,//0.2
		0.6179,	0.6217,	0.6255,	0.6293,	0.6331,	0.6368,	0.6404,	0.6443,	0.648,	0.6517,//0.3
		0.6554,	0.6591,	0.6628,	0.6664,	0.67,	0.6736,	0.6772,	0.6808,	0.6844,	0.6879,//0.4
		0.6915,	0.695,	0.6985,	0.7019,	0.7054,	0.7088,	0.7123,	0.7157,	0.719,	0.7224,//0.5
		0.7257,	0.7291,	0.7324,	0.7357,	0.7389,	0.7422,	0.7454,	0.7486,	0.7517,	0.7549,//0.6
		0.758,	0.7611,	0.7642,	0.7673,	0.7703,	0.7734,	0.7764,	0.7794,	0.7823,	0.7852,//0.7
		0.7881,	0.791,	0.7939,	0.7967,	0.7995,	0.8023,	0.8051,	0.8078,	0.8106,	0.8133,//0.8
		0.8159,	0.8186,	0.8212,	0.8238,	0.8264,	0.8289,	0.8355,	0.834,	0.8365,	0.8389,//0.9
		0.8413,	0.8438,	0.8461,	0.8485,	0.8508,	0.8531,	0.8554,	0.8577,	0.8599,	0.8621,//1.0
		0.8643,	0.8665,	0.8686,	0.8708,	0.8729,	0.8749,	0.877,	0.879,	0.881,	0.883,//1.1
		0.8849,	0.8869,	0.8888,	0.8907,	0.8925,	0.8944,	0.8962,	0.898,	0.8997,	0.9015,//1.2
		0.9032,	0.9049,	0.9066,	0.9082,	0.9099,	0.9115,	0.9131,	0.9147,	0.9162,	0.9177,//1.3
		0.9192,	0.9207,	0.9222,	0.9236,	0.9251,	0.9265,	0.9279,	0.9292,	0.9306,	0.9319,//1.4
		0.9332,	0.9345,	0.9357,	0.937,	0.9382,	0.9394,	0.9406,	0.9418,	0.943,	0.9441,//1.5
		0.9452,	0.9463,	0.9474,	0.9484,	0.9495,	0.9505,	0.9515,	0.9525,	0.9535,	0.9535,//1.6
		0.9554,	0.9564,	0.9573,	0.9582,	0.9591,	0.9599,	0.9608,	0.9616,	0.9625,	0.9633,//1.7
		0.9641,	0.9648,	0.9656,	0.9664,	0.9672,	0.9678,	0.9686,	0.9693,	0.97,	0.9706,//1.8
		0.9713,	0.9719,	0.9726,	0.9732,	0.9738,	0.9744,	0.975,	0.9756,	0.9762,	0.9767,//1.9
		0.9772,	0.9778,	0.9783,	0.9788,	0.9793,	0.9798,	0.9803,	0.9808,	0.9812,	0.9817,//2.0
		0.9821,	0.9826,	0.983,	0.9834,	0.9838,	0.9842,	0.9846,	0.985,	0.9854,	0.9857,//2.1
		0.9861,	0.9864,	0.9868,	0.9871,	0.9874,	0.9878,	0.9881,	0.9884,	0.9887,	0.989,//2.2
		0.9893,	0.9896,	0.9898,	0.9901,	0.9904,	0.9906,	0.9909,	0.9911,	0.9913,	0.9916,//2.3
		0.9918,	0.992,	0.9922,	0.9925,	0.9927,	0.9929,	0.9931,	0.9932,	0.9934,	0.9936,//2.4
		0.9938,	0.994,	0.9941,	0.9943,	0.9945,	0.9946,	0.9948,	0.9949,	0.9951,	0.9952,//2.5
		0.9953,	0.9955,	0.9956,	0.9957,	0.9959,	0.996,	0.9961,	0.9962,	0.9963,	0.9964,//2.6
		0.9965,	0.9966,	0.9967,	0.9968,	0.9969,	0.997,	0.9971,	0.9972,	0.9973,	0.9974,//2.7
		0.9974,	0.9975,	0.9976,	0.9977,	0.9977,	0.9978,	0.9979,	0.9979,	0.998,	0.9981,//2.8
		0.9981,	0.9982,	0.9982,	0.9983,	0.9984,	0.9984,	0.9985,	0.9985,	0.9986,	0.9986,//2.9
		0,9987, 0.999,  0.9993, 0.9995, 0.9997, 0.9998, 0.9998, 0.9999, 0.9999, 1.0000//3.0
	};//N(0,1) Gaussian distribution table
		if(std==0)
			return false;//all values are the same
		double std_value = fabs((value-mean)/std);//standardlized value
		int f_dig = (int)(std_value*10);
		if(f_dig>30)//larger than 30
			return true;
		int s_dig = (int)(std_value*100) - f_dig*10;
		double ci = (1-gaussian_table[f_dig*10+s_dig])*2;
		return ci*num<threshold;//if true is returned, the outlier can be discarded
	}

	/*Given the mean vector and covariance. Compute the value of Normal distribution
	We assume correlations among different dimensions are zero*/
	template<typename T>
	inline double GaussianDistribution(const T *mean, const T *std, size_t N, const T *value, bool pre = true)
	{
		//We prefer N=1,2,i.e, mono, bivariate case
		if(N==1)//1D case
		{
			double pre_value(1);
			if(pre)
				pre_value=1/(sqrt(2*PI)*std[0]);
			double pro_value = 
				pre_value*exp(-0.5*
				(value[0]-mean[0])*(value[0]-mean[0])/(std[0]*std[0]));
			return pro_value;
		}
		else if(N==2)//2D case
		{
			double pre_value(1);
			if(pre)
				pre_value=1/(2*PI*std[0]*std[1]);
			double pro_value = 
				pre_value*exp(-0.5*
				((value[0]-mean[0])*(value[0]-mean[0])/(std[0]*std[0])
				+(value[1]-mean[1])*(value[1]-mean[1])/(std[1]*std[1])));
			return pro_value;
		}else//Multivariate case
		{
			double determinant(1),post_value(0);//the sqrt determinant of cov matrix
			for(size_t i = 0; i<N; ++i)
			{
				determinant *= std[i];
				post_value += (value[i]-mean[i])*(value[i]-mean[i])/((double)std[i]*std[i]);//convert to double
			}
			double pre_value(1);
			if(pre)
			{
				pre_value = 
					1/(pow(2*PI,N/2.0)*determinant);
			}
			double prob_value = pre_value*exp(-0.5*post_value);
			return prob_value;
		}
	}

	/*Given the mapping function and coordinates in mapped system, return the coordinates in original coordinate system*/
	inline void MapCoordinates(const int org[2], const int map[2], const int value_map[2], int value_org[2])
	{
		int transform[]={map[0]-org[0],map[1]-org[1]};
		value_org[0]=value_map[0]-transform[0];
		value_org[1]=value_map[1]-transform[1];
	}

	/*Produce a linear function and give a new value f(x) based on x*/
	template<typename T>
	inline T LinearFunc(T x1, T x2, T y1, T y2, T x)
	{
		return (y2-y1)*(x-x1)/(x2-x1)+y1;
	}

	/*Compute L2 norm of a vector*/
	template<typename T>
	inline double L2Norm(T *vec, int N, bool sqrt)
	{
		double sum = 0;
		for(int i=0; i<N; ++i)
		{
			sum+= vec[i]*vec[i];
		}
		if(sqrt)
			return std::sqrt(sum);
		else
			return sum;
	}

	/*TOPK algorithm and returns sorted index*/
	template<typename T>
	void TOPK(T arr[],int index[],int length,int k)//find the top k strongest elements
	{
		_topk(arr,index,0,length-1,k);
	}

	/*recursive topk algorithm based on quick sort*/
	template<typename T>
	static void _topk(T arr[],int index[],int start,int end,int k)//from start to end, we aim to find top k
	{
		if(start < end) {
			T mid = arr[rand()%(end-start+1) + start];
			int i = start;
			int j = end;
			do{
				while(arr[i]>=mid) i++;//we maintian the max number as much as possible
				while(arr[j]<mid) j--;//we bring the equal one to the front
				if(i<j&&arr[i]!=arr[j])
					swap(arr[i],arr[j],index[i],index[j]);
				//else
					//break;
			}while(i<j);
			//i and j meets at the mid point
			//j (not included)以后是比mid小的数
			int num=j-start+1;//number of larger values
			if(num==k)//inclusive, equal k strongest
			{
				return;//we have done the job
			}
			else if(num>k)// more than k, we need to do further seperation
			{
				//We are not sure whether we have reduced the size of the problem
				if(j==end)//The size of the problem is not reduced
				{
					//Do quick sort
					quicksort(arr,index,start,j);
				}
				else
					_topk(arr,index,start,j,k);
			}
			else //less than k
			{
				int rest=k-num;
				_topk(arr,index,j+1,end,rest);//we have to sort rest number of points
			}
		}	
	}

	/*swap two values*/
	template<typename T>
	inline void swap(T &a,T &b,int &x,int &y)
	{
		T temp=a; int t=x;
		a=b; x=y;
		b=temp; y=t;
	}

	/*quick sort*/
	template<typename T>
	void quicksort(T arr[],int index[],int start,int end)
	{
		if(start < end) {
			T mid = arr[rand()%(end-start+1) + start];
			int i = start;
			int j = end;
			do{
				while(arr[i]>mid) i++;//we maintian the max number as much as possible
				while(arr[j]<mid) j--;//we bring the equal one to the front
				if(i<j&&arr[i]!=arr[j])
					swap(arr[i],arr[j],index[i],index[j]);
				else
					break;
			}while(i<j);
			quicksort(arr,index,start,j-1);//reduce the size of the problem
			quicksort(arr,index,j+1,end);
		}
	}

	/*Epanechnikov Kernel (Currently, we only support N=1(circle) and N=2 (sphere))*/
	template<typename T>
	inline double Epanechnikov_Kernel(T x[], size_t N)//N-Volumn of N-dimensional sphere (The surface (N+1)-ball)
	{
		assert(N==1||N==2);//only support circle or sphere
		double norm2 = 0;
		for(int i = 0; i<N; ++i)
		{
			norm2+=x[i]*x[i];
		}
		if(sqrt(norm2)>=1)
			return 0;
		else
		{
			double N_volumn = 0;//volumn of unit N-sphere
			if(N==1)
			{
				N_volumn = 2*PI;//2*pi*R
			}else if(N==2)
			{
				N_volumn = 4*PI;//4*pi*R^2
			}

			double val = 0.5/N_volumn*(N+2)*(1-norm2);
			return val;
		}
	}

	/*Kronecker delta function*/
	template<typename T>
	inline int Kronecker_delta(T i)
	{
		if(abs(i) <= DBL_EPSILON)
			return 1;
		else
			return 0;
	}

	/*Round*/
	template<typename T>
	inline int Round(T x)
	{
		return (int)(x+0.5);
	}

	/*Create 2D array dynamically*/
	inline void** _createArray2D(int ncols, int nrows, int nbytes)
	{
		char **tt;
		int i;

		tt = (char **) malloc(nrows * sizeof(void *) +
                        ncols * nrows * nbytes);
		if (tt == NULL)
		{
			fprintf(stderr,"(createArray2D) Out of memory");
			system("abort");
			system("pause");
			return NULL;
		}
	
		for (i = 0 ; i < nrows ; i++)
			tt[i] = ((char *) tt) + (nrows * sizeof(void *) +//assign address to the nrows 1D pointer
								i * ncols * nbytes);

		return((void **) tt);
	}


	/*
	 * Lower tail quantile for standard normal distribution function.
	 *
	 * This function returns an approximation of the inverse cumulative
	 * standard normal distribution function.  I.e., given P, it returns
	 * an approximation to the X satisfying P = Pr{Z <= X} where Z is a
	 * random variable from the standard normal distribution.
	 *
	 * The algorithm uses a minimax approximation by rational functions
	 * and the result has a relative error whose absolute value is less
	 * than 1.15e-9.
	 *
	 * Author:      Peter John Acklam
	 * Time-stamp:  2002-06-09 18:45:44 +0200
	 * E-mail:      jacklam@math.uio.no
	 * WWW URL:     http://www.math.uio.no/~jacklam
	 *
	 * C implementation adapted from Peter's Perl version
	 */

	/* Coefficients in rational approximations. */
	static const double a[] =
	{
		-3.969683028665376e+01,
		 2.209460984245205e+02,
		-2.759285104469687e+02,
		 1.383577518672690e+02,
		-3.066479806614716e+01,
		 2.506628277459239e+00
	};

	static const double b[] =
	{
		-5.447609879822406e+01,
		 1.615858368580409e+02,
		-1.556989798598866e+02,
		 6.680131188771972e+01,
		-1.328068155288572e+01
	};

	static const double c[] =
	{
		-7.784894002430293e-03,
		-3.223964580411365e-01,
		-2.400758277161838e+00,
		-2.549732539343734e+00,
		 4.374664141464968e+00,
		 2.938163982698783e+00
	};

	static const double d[] =
	{
		7.784695709041462e-03,
		3.224671290700398e-01,
		2.445134137142996e+00,
		3.754408661907416e+00
	};

	// All the functions should be declared as inline functions if they are defined in .h file
	inline double ltqnorm(double p)
	{
		double q, r;

		errno = 0;

		if (p < 0 || p > 1)
		{
			errno = EDOM;
			return 0.0;
		}
		else if (p == 0)
		{
			errno = ERANGE;
			return -HUGE_VAL /* minus "infinity" */;
		}
		else if (p == 1)
		{
			errno = ERANGE;
			return HUGE_VAL /* "infinity" */;
		}
		else if (p < LOW)
		{
			/* Rational approximation for lower region */
			q = sqrt(-2*log(p));
			return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
				((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
		}
		else if (p > HIGH)
		{
			/* Rational approximation for upper region */
			q  = sqrt(-2*log(1-p));
			return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
				((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
		}
		else
		{
			/* Rational approximation for central region */
    		q = p - 0.5;
    		r = q*q;
			return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
				(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
		}
	}

	inline bool isFiniteNumber(double num)
	{
		return (num <= DBL_MAX && num >= -DBL_MAX);
	}

	/*Draw a random number from gaussian distribution with mu and sigma N(mu,sigma)*/
	inline double randn(double mu, double sigma)
	{
		if (sigma == 0)
		{
			return mu;
		}
		double uniform = 0;
		do 
		{
			uniform = rand() / double(RAND_MAX); //uniform distribution in [0,1]
		} while (uniform ==0.0 || uniform ==1.0);
		double std_gaussian = ltqnorm(uniform);
		assert(isFiniteNumber(std_gaussian));
		double gaussian = std_gaussian * sigma + mu;
		return gaussian;
	}

	/*sign function*/
	template<typename T>
	inline int sgn(T val)
	{
		return int(T(0) < val) - (val < T(0));
	}

	/*POW2*/
	template<typename T>
	inline T pow2(T val)
	{
		return val * val;
	}
}
