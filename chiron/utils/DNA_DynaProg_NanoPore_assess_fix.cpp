#include <string>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;


//--------- base_name -----------//__110830__//
void getBaseName(string &in,string &out,char slash,char dot)
{
	int i,j;
	int len=(int)in.length();
	for(i=len-1;i>=0;i--)
	{
		if(in[i]==slash)break;
	}
	i++;
	for(j=len-1;j>=0;j--)
	{
		if(in[j]==dot)break;
	}
	if(j==-1)j=len;
	out=in.substr(i,j-i);
}
void getRootName(string &in,string &out,char slash)
{
	int i;
	int len=(int)in.length();
	for(i=len-1;i>=0;i--)
	{
		if(in[i]==slash)break;
	}
	if(i<=0)out=".";
	else out=in.substr(0,i);
}

//=========== Read FASTA sequence ==========//
int Read_FASTA_SEQRES(string &infile,string &seqres,int skip=1) //->from .fasta file
{
	ifstream fin;
	string buf,temp;
	//read
	fin.open(infile.c_str(), ios::in);
	if(fin.fail()!=0)
	{
		fprintf(stderr,"no such file! %s \n",infile.c_str());
		return -1;
	}
	//skip
	int i;
	for(i=0;i<skip;i++)
	{
		if(!getline(fin,buf,'\n'))
		{
			fprintf(stderr,"file bad! %s \n",infile.c_str());
			return -1;
		}
	}
	//process
	temp="";
	for(;;)
	{
		if(!getline(fin,buf,'\n'))break;
		temp+=buf;
	}
	seqres=temp;
	//return
	return (int)seqres.length();
}


//=========== blosum calculate ============//
//=================//
//--Ori_BLOSUM----//
//===============//
int Ori_BLOSUM_62[21][21]={
{  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -5 },  //A
{ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -5 },  //R
{ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, -5 },  //N
{ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, -5 },  //D
{  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -5 },  //C
{ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, -5 },  //Q
{ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, -5 },  //E
{  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -5 },  //G
{ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, -5 },  //H
{ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -5 },  //I
{ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -5 },  //L
{ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, -5 },  //K
{ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -5 },  //M
{ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -5 },  //F
{ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -5 },  //P
{  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, -5 },  //S
{  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -5 },  //T
{ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -5 },  //W
{ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -5 },  //Y
{  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -5 },  //V
{ -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5 }}; //Z
// A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   Z

int Ori_BLOSUM_45[21][21]={
{  5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -2, -2,  0, -5 },  //A
{ -2,  7,  0, -1, -3,  1,  0, -2,  0, -3, -2,  3, -1, -2, -2, -1, -1, -2, -1, -2, -5 },  //R
{ -1,  0,  6,  2, -2,  0,  0,  0,  1, -2, -3,  0, -2, -2, -2,  1,  0, -4, -2, -3, -5 },  //N
{ -2, -1,  2,  7, -3,  0,  2, -1,  0, -4, -3,  0, -3, -4, -1,  0, -1, -4, -2, -3, -5 },  //D
{ -1, -3, -2, -3, 12, -3, -3, -3, -3, -3, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -5 },  //C
{ -1,  1,  0,  0, -3,  6,  2, -2,  1, -2, -2,  1,  0, -4, -1,  0, -1, -2, -1, -3, -5 },  //Q
{ -1,  0,  0,  2, -3,  2,  6, -2,  0, -3, -2,  1, -2, -3,  0,  0, -1, -3, -2, -3, -5 },  //E
{  0, -2,  0, -1, -3, -2, -2,  7, -2, -4, -3, -2, -2, -3, -2,  0, -2, -2, -3, -3, -5 },  //G
{ -2,  0,  1,  0, -3,  1,  0, -2, 10, -3, -2, -1,  0, -2, -2, -1, -2, -3,  2, -3, -5 },  //H
{ -1, -3, -2, -4, -3, -2, -3, -4, -3,  5,  2, -3,  2,  0, -2, -2, -1, -2,  0,  3, -5 },  //I
{ -1, -2, -3, -3, -2, -2, -2, -3, -2,  2,  5, -3,  2,  1, -3, -3, -1, -2,  0,  1, -5 },  //L
{ -1,  3,  0,  0, -3,  1,  1, -2, -1, -3, -3,  5, -1, -3, -1, -1, -1, -2, -1, -2, -5 },  //K
{ -1, -1, -2, -3, -2,  0, -2, -2,  0,  2,  2, -1,  6,  0, -2, -2, -1, -2,  0,  1, -5 },  //M
{ -2, -2, -2, -4, -2, -4, -3, -3, -2,  0,  1, -3,  0,  8, -3, -2, -1,  1,  3,  0, -5 },  //F
{ -1, -2, -2, -1, -4, -1,  0, -2, -2, -2, -3, -1, -2, -3,  9, -1, -1, -3, -3, -3, -5 },  //P
{  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -3, -1, -2, -2, -1,  4,  2, -4, -2, -1, -5 },  //S
{  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -1, -1,  2,  5, -3, -1,  0, -5 },  //T
{ -2, -2, -4, -4, -5, -2, -3, -2, -3, -2, -2, -2, -2,  1, -3, -4, -3, 15,  3, -3, -5 },  //W
{ -2, -1, -2, -2, -3, -1, -2, -3,  2,  0,  0, -1,  0,  3, -3, -2, -1,  3,  8, -1, -5 },  //Y
{  0, -2, -3, -3, -1, -3, -3, -3, -3,  3,  1, -2,  1,  0, -3, -1,  0, -3, -1,  5, -5 },  //V
{ -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5 }}; //Z
// A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   Z  

int Ori_BLOSUM_80[21][21]={
{  5, -2, -2, -2, -1, -1, -1,  0, -2, -2, -2, -1, -1, -3, -1,  1,  0, -3, -2,  0, -5 },  //A
{ -2,  6, -1, -2, -4,  1, -1, -3,  0, -3, -3,  2, -2, -4, -2, -1, -1, -4, -3, -3, -5 },  //R
{ -2, -1,  6,  1, -3,  0, -1, -1,  0, -4, -4,  0, -3, -4, -3,  0,  0, -4, -3, -4, -5 },  //N
{ -2, -2,  1,  6, -4, -1,  1, -2, -2, -4, -5, -1, -4, -4, -2, -1, -1, -6, -4, -4, -5 },  //D
{ -1, -4, -3, -4,  9, -4, -5, -4, -4, -2, -2, -4, -2, -3, -4, -2, -1, -3, -3, -1, -5 },  //C
{ -1,  1,  0, -1, -4,  6,  2, -2,  1, -3, -3,  1,  0, -4, -2,  0, -1, -3, -2, -3, -5 },  //Q
{ -1, -1, -1,  1, -5,  2,  6, -3,  0, -4, -4,  1, -2, -4, -2,  0, -1, -4, -3, -3, -5 },  //E
{  0, -3, -1, -2, -4, -2, -3,  6, -3, -5, -4, -2, -4, -4, -3, -1, -2, -4, -4, -4, -5 },  //G
{ -2,  0,  0, -2, -4,  1,  0, -3,  8, -4, -3, -1, -2, -2, -3, -1, -2, -3,  2, -4, -5 },  //H
{ -2, -3, -4, -4, -2, -3, -4, -5, -4,  5,  1, -3,  1, -1, -4, -3, -1, -3, -2,  3, -5 },  //I
{ -2, -3, -4, -5, -2, -3, -4, -4, -3,  1,  4, -3,  2,  0, -3, -3, -2, -2, -2,  1, -5 },  //L
{ -1,  2,  0, -1, -4,  1,  1, -2, -1, -3, -3,  5, -2, -4, -1, -1, -1, -4, -3, -3, -5 },  //K
{ -1, -2, -3, -4, -2,  0, -2, -4, -2,  1,  2, -2,  6,  0, -3, -2, -1, -2, -2,  1, -5 },  //M
{ -3, -4, -4, -4, -3, -4, -4, -4, -2, -1,  0, -4,  0,  6, -4, -3, -2,  0,  3, -1, -5 },  //F
{ -1, -2, -3, -2, -4, -2, -2, -3, -3, -4, -3, -1, -3, -4,  8, -1, -2, -5, -4, -3, -5 },  //P
{  1, -1,  0, -1, -2,  0,  0, -1, -1, -3, -3, -1, -2, -3, -1,  5,  1, -4, -2, -2, -5 },  //S
{  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -2, -1, -1, -2, -2,  1,  5, -4, -2,  0, -5 },  //T
{ -3, -4, -4, -6, -3, -3, -4, -4, -3, -3, -2, -4, -2,  0, -5, -4, -4, 11,  2, -3, -5 },  //W
{ -2, -3, -3, -4, -3, -2, -3, -4,  2, -2, -2, -3, -2,  3, -4, -2, -2,  2,  7, -2, -5 },  //Y
{  0, -3, -4, -4, -1, -3, -3, -4, -4,  3,  1, -3,  1, -1, -3, -2,  0, -3, -2,  4, -5 },  //V
{ -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5 }}; //Z
// A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   Z  


//BLOSUM_Mapping//--------------ARNDCQEGHILKMFPSTWYVZ
int Blo_AA_Map_WS[21]=
{ 0,19, 4, 3, 6, 13,7, 8, 9, 17,11,10,12,2, 18,14,5, 1, 15,16,20};
//A  V  C  D  E  F  G  H  I  W  K  L  M  N  Y  P  Q  R   S  T  Z
//0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17  18 19 20
//Ori_Mapping//-----------------AVCDEFGHIWKLMNYPQRSTZ
int Ori_AA_Map_WS[26]=
{ 0,20,2,3,4,5,6,7,8,20,10,11,12,13,20,15,16,17,18,19,20, 1, 9,20,14,20};
// A B C D E F G H I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z
// 0 1 2 3 4 5 6 7 8  9 10 11 12 14 14 15 16 17 18 19 20 21 22 23 24 25

//------ calculate -------//
int BLOSUM62_Calc(char a,char b)
{
	int ii,jj;
	if(a<'A' || a>'Z')a='Z';
	ii=Blo_AA_Map_WS[Ori_AA_Map_WS[a-'A']];
	if(b<'A' || b>'Z')b='Z';
	jj=Blo_AA_Map_WS[Ori_AA_Map_WS[b-'A']];
	return Ori_BLOSUM_62[ii][jj];
}
int BLOSUM45_Calc(char a,char b)
{
	int ii,jj;
	if(a<'A' || a>'Z')a='Z';
	ii=Blo_AA_Map_WS[Ori_AA_Map_WS[a-'A']];
	if(b<'A' || b>'Z')b='Z';
	jj=Blo_AA_Map_WS[Ori_AA_Map_WS[b-'A']];
	return Ori_BLOSUM_45[ii][jj];
}
int BLOSUM80_Calc(char a,char b)
{
	int ii,jj;
	if(a<'A' || a>'Z')a='Z';
	ii=Blo_AA_Map_WS[Ori_AA_Map_WS[a-'A']];
	if(b<'A' || b>'Z')b='Z';
	jj=Blo_AA_Map_WS[Ori_AA_Map_WS[b-'A']];
	return Ori_BLOSUM_80[ii][jj];
}


//=================== NanoPore DynaProg ===============//
//-> ATCG to digit
//-> see below link for base pairing:
//       http://www.biology-pages.info/B/BasePairing.html
//-> see below link for molecular weight:
//       http://www.genomics.agilent.com/files/Mobio/Nucleic%20Acids_Sizes_and_Molecular_Weights_2pgs.pdf
int DNA_To_Int(char c)
{
	switch(c)
	{
		case 'A': return -2;
		case 'T': return -1;
		case 'C': return  1;
		case 'G': return  2;
		return 0;
	}
}

//-> calculate NanoPore DNA similarity
int NanoPore_DNA_Similarity(char a,char b)
{
	//-> transform to int
	int a_=DNA_To_Int(a);
	int b_=DNA_To_Int(b);
	//-> check bad
	if(a_==0 || b_==0)return -2;
	//-> if same, return hightest similarity
	if(a_==b_)return 1;
	if(abs(a_)==abs(b_))return 0;
	return -2;
}

//-> calculate 5mer similarity 
double NanoPore_Calc(const char *a, const char *b, int i,int j,int n1,int n2)
{
	int scale;
	int hwin=2;
	int score=0;
	int count=0;
	for(int k=-hwin;k<=hwin;k++)
	{
		count++;
		int ii=i+k;
		int jj=j+k;
		if(ii<0 || jj<0)continue;
		if(ii>=n1 || jj>=n2)break;
		if(k==0)scale=4;
		else if(abs(k)==1)scale=2;
		else scale=1;
		score+=scale*NanoPore_DNA_Similarity(a[ii],b[jj]);
	}
	return (double)(0.5*score/count);
}


//-------- DNA calc ------//
int DNA_Calc(char a,char b)
{
	int mat=1;
	int mis=-2;
	if(a!='A' && a!='T' && a!='C' && a!='G')return mis;
	if(b!='A' && b!='T' && b!='C' && b!='G')return mis;
	if(a==b)return mat;
	else return mis;
}


//---------- dynamic programming ----------//
int WWW_Advance_Align_Dyna_Prog_Double(int n1,int n2,const vector<double> &score,
								   double GAP_OPEN1,double GAP_EXT1,double GAP_OPEN2,double GAP_EXT2,
								   double GAP_HEAD1,double GAP_TAIL1,double GAP_HEAD2,double GAP_TAIL2,
								   vector<pair<int,int> > & alignment,double &ali_sco)
{
	int i,j;
	//input
	int m = n1 + 1;  // +1 to account for the extra row,col in
	int n = n2 + 1;  // the DP matrices corresponding to gaps
	int DP_maximal=n;
	int IN_maximal=n2;
	//const value
	const int _H_  = 0;
	const int _S_  = 1;
	const int _V_  = 2;

	//create D and M
	vector <int> D[3];      // the path (directions) matrix
	vector <double> M[3];   // the current scores (values) matrix
	//resize(m,n)
	for (i = 0; i < 3; ++i) 
	{
		D[i].resize(m*n);
		M[i].resize(m*n);
	}
	//init()
	double WS_MIN=-1000000;
	D[_S_][0*DP_maximal+ 0] = -1;
	D[_H_][0*DP_maximal+ 0] = -1;
	D[_V_][0*DP_maximal+ 0] = -1;
	M[_S_][0*DP_maximal+ 0] = 0;
	M[_H_][0*DP_maximal+ 0] = WS_MIN;
	M[_V_][0*DP_maximal+ 0] = WS_MIN;
	for (i = 1; i < m; i++) 
	{
		D[_S_][i*DP_maximal+ 0] = _V_;
		D[_H_][i*DP_maximal+ 0] = _V_;
		D[_V_][i*DP_maximal+ 0] = _V_;
		M[_S_][i*DP_maximal+ 0] = WS_MIN;
		M[_H_][i*DP_maximal+ 0] = WS_MIN;
		M[_V_][i*DP_maximal+ 0] = i*GAP_HEAD1; //-(Params::GAP_OPEN + (i-1)*Params::GAP_EXT);
	}
	for (j = 1; j < n; j++) 
	{
		D[_S_][0*DP_maximal+ j] = _H_;
		D[_H_][0*DP_maximal+ j] = _H_;
		D[_V_][0*DP_maximal+ j] = _H_;
		M[_S_][0*DP_maximal+ j] = WS_MIN;
		M[_H_][0*DP_maximal+ j] = j*GAP_HEAD2; //-(Params::GAP_OPEN + (j-1)*Params::GAP_EXT);
		M[_V_][0*DP_maximal+ j] = WS_MIN;
	}
	//fill(firstSeq, secondSeq, distFunc);
	double gap_open;
	double gap_ext;
	double v1,v2,v3;
	double dist;
	for (i = 1; i < m; i++) 
	{
		for (j = 1; j < n; j++) 
		{
			//condition upper
			if(j==n-1)
			{
				gap_open=GAP_TAIL1;
				gap_ext=GAP_TAIL1;
			}
			else
			{
				gap_open=GAP_OPEN1;
				gap_ext=GAP_EXT1;
			}
			v1 = M[_V_][(i-1)*DP_maximal+ j] + gap_ext;
			v2 = M[_S_][(i-1)*DP_maximal+ j] + gap_open;
			v3 = M[_H_][(i-1)*DP_maximal+ j] + gap_open;
			M[_V_][i*DP_maximal+ j] = std::max(v1, std::max(v2, v3));
			if (M[_V_][i*DP_maximal+ j] == v1) D[_V_][i*DP_maximal+ j] = _V_;
			else if(M[_V_][i*DP_maximal+ j] == v2) D[_V_][i*DP_maximal+ j] = _S_;
			else D[_V_][i*DP_maximal+ j] = _H_;
			//condition left
			if(i==m-1)
			{
				gap_open=GAP_TAIL2;
				gap_ext=GAP_TAIL2;
			}
			else
			{
				gap_open=GAP_OPEN2;
				gap_ext=GAP_EXT2;
			}
			v1 = M[_H_][i*DP_maximal+ j-1] + gap_ext;
			v2 = M[_S_][i*DP_maximal+ j-1] + gap_open;
			v3 = M[_V_][i*DP_maximal+ j-1] + gap_open;
			M[_H_][i*DP_maximal+ j] = std::max(v1, std::max(v2, v3));
			if (M[_H_][i*DP_maximal+ j] == v1) D[_H_][i*DP_maximal+ j] = _H_;
			else if(M[_H_][i*DP_maximal+ j] == v2) D[_H_][i*DP_maximal+ j] = _S_;
			else D[_H_][i*DP_maximal+ j] = _V_;
			//condition diag
			dist = score.at((i-1)*IN_maximal+ j-1);  //Params::K - distFunc(firstSeq[i-1], secondSeq[j-1]);
			v1 = M[_V_][(i-1)*DP_maximal+ j-1] + dist;
			v2 = M[_H_][(i-1)*DP_maximal+ j-1] + dist;
			v3 = M[_S_][(i-1)*DP_maximal+ j-1] + dist;
			M[_S_][i*DP_maximal+ j] = std::max(v1, std::max(v2, v3));
			if (M[_S_][i*DP_maximal+ j] == v3) D[_S_][i*DP_maximal+ j] = _S_;
			else if (M[_S_][i*DP_maximal+ j] == v1) D[_S_][i*DP_maximal+ j] = _V_;
			else D[_S_][i*DP_maximal+ j] = _H_;
		}
	}
	//build(ali, firstSeq, secondSeq, distFunc);
	i = m-1;
	j = n-1;
	v1=M[_V_][i*DP_maximal+ j];
	v2=M[_H_][i*DP_maximal+ j];
	v3=M[_S_][i*DP_maximal+ j];
	double maximal = std::max(v1, std::max(v2, v3));
	int k = -1;
	if(v3==maximal)k = _S_;
	else if(v2==maximal)k = _H_;
	else k = _V_;
	//trace_back
	alignment.clear();
	int count = 0;
	int matches = 0;
	int cur_case=k;
	int pre_case;
	for(;;)
	{
		if(i==0||j==0)break;
		pre_case=D[cur_case][i*DP_maximal+ j];
		switch (cur_case)
		{
			case _S_:
				alignment.push_back(pair<int,int>(i,j)); 
				i--;
				j--;
				++matches;
				break;
			case _V_:
				alignment.push_back(pair<int,int>(i,-j)); 
				i--;
				break;
			case _H_:
				alignment.push_back(pair<int,int>(-i,j)); 
				j--;
				break;
			default:
				cout << "ERROR!! -> advance_global: invalid direction D[" << k << "](" << i << ", " << j << ") = " 
				<< D[k][i*DP_maximal+ j] << endl;
				exit(-1);
		}
		cur_case=pre_case;
		count++;
	}
	while (j> 0) alignment.push_back(pair<int,int>(-i,j)),j--;
	while (i> 0) alignment.push_back(pair<int,int>(i,0)), i--;
	reverse(alignment.begin(), alignment.end());
	ali_sco=maximal;
	return matches;
}

//============= Exttract Alignment ========//
//-> get mapping alignment detail
double process_oriami_record_simp(const char *seq_,const char *ami_,
	vector<pair<int,int> > &WWW_alignment,int &matchs)
{
	int i,j;
	int n1,n2;
	//--[1]dynamic_programming	
	n1=(int)strlen(seq_);
	n2=(int)strlen(ami_);
	vector <double> WWW_score;
	WWW_score.resize(n1*n2);
	for(i=0;i<n1;i++)
	{
		for(j=0;j<n2;j++)
		{
//			WWW_score[i*n2+j]=BLOSUM62_Calc(seq_[i],ami_[j]);
//			WWW_score[i*n2+j]=DNA_Calc(seq_[i],ami_[j]);
			WWW_score[i*n2+j]=NanoPore_Calc(seq_,ami_,i,j,n1,n2);
		}
	}
	double sco;
	matchs=WWW_Advance_Align_Dyna_Prog_Double(n1,n2,WWW_score,-2,-1,-2,-1,0,0,0,0,
		WWW_alignment,sco);
	return sco;
}


//------- given Ali1 return AliPair ------//
void Ali1_To_AliPair(int n1,int n2,vector <int> &ali1,
	vector<pair<int,int> > & alignment_out)
{
	//init
	alignment_out.clear();
	//start
	int i,j;
	int ii,jj;
	int wlen;
	int pre_ii=0;
	int pre_jj=0;
	for(i=1;i<=n1;i++)
	{
		ii=i;
		jj=ali1[i-1];  //ali1 starts from 0, correspondence also from 0
		if(jj==-1)
		{
			continue;
		}
		else
		{
			jj++;
			//previous_path
			wlen=ii-pre_ii;
			for(j=1;j<wlen;j++)
			{
				pre_ii++;
				alignment_out.push_back (pair<int,int>(pre_ii, -pre_jj)); //Ix
			}
			wlen=jj-pre_jj;
			for(j=1;j<wlen;j++)
			{
				pre_jj++;
				alignment_out.push_back (pair<int,int>(-pre_ii, pre_jj)); //Iy
			}
			//current_path
			alignment_out.push_back (pair<int,int>(ii, jj)); //Match
			//update
			pre_ii=ii;
			pre_jj=jj;
		}
	}
	//termi
	pre_ii++;
	for(i=pre_ii;i<=n1;i++)alignment_out.push_back (pair<int,int>(i, -pre_jj)); //Ix
	pre_jj++;
	for(i=pre_jj;i<=n2;i++)alignment_out.push_back (pair<int,int>(-n1, i));  //Iy
}

//------- given AliPair return Ali1 and Ali2 ------//
void AliPair_To_Ali1_Ali2(int n1,int n2,
	vector<pair<int,int> > & alignment_in,
	vector <int> &ali1,vector <int> &ali2)
{
	//init
	ali1.resize(n1);
	ali2.resize(n2);
	for(int i=0;i<n1;i++)ali1[i]=-1;
	for(int i=0;i<n2;i++)ali2[i]=-1; 
	//proc
	for(int i=0;i<(int)alignment_in.size();i++)
	{
		int ii=alignment_in[i].first;
		int jj=alignment_in[i].second;
		if(ii>0 && jj>0)
		{
			ali1[ii-1]=jj-1;
			ali2[jj-1]=ii-1;
		}
	}
}

//-------- Ali_To_Cor -------------//
int Ali_To_Cor(vector <int> &ali2, vector <vector <int> > &AFP_Cor)
{
	int i,k;
	int num;
	int ii,jj;
	int count;
	int isFirst;
	int isLast;
	int type;
	int head1,head2;
	int index;

	//init
	count=-999999;
	num=0;
	head1=-1;
	head2=-1;
	isLast=0;
	isFirst=1;
	type=0;
	ii=-1;
	jj=-1;
	int moln2=(int)ali2.size();
	int thres=0;
	AFP_Cor.clear();
	for(i=0;i<moln2;i++)
	{
		if(ali2[i]==-1) //purely blank
		{
			if(isFirst==0)
			{
				if(count>=thres) 
				{
					vector <int> tmp_rec;
					tmp_rec.push_back(head1);
					tmp_rec.push_back(head2);
					tmp_rec.push_back(count);
					AFP_Cor.push_back(tmp_rec);
					num+=count;
				}
				count=0;
				isFirst=1;
			}
			continue;
		}

		if(isFirst==1)
		{
ws_init:
			isFirst=0;
			ii=ali2[i];
			type=1; // >0 mode
			jj=i;
			count=1;
			head1=ii;
			head2=jj;
			continue;
		}
		if(i==jj+1&&ali2[i]==ii+1)
		{
			ii=ali2[i];
			jj=i;
			count++;
			continue;
		}

ws_end:
		if(count>=thres) 
		{
			vector <int> tmp_rec;
			tmp_rec.push_back(head1);
			tmp_rec.push_back(head2);
			tmp_rec.push_back(count);
			AFP_Cor.push_back(tmp_rec);
			num+=count;
		}

		if(isLast==1)goto end;
		else goto ws_init;
	}

	if(count==999999)goto end;
	isLast=1;
	goto ws_end;
end:
	return num;
}

//--------- head tail fix ----------//
int Head_Tail_Fix(string &str1,string &str2,vector <vector <int> > &AFP_Cor,
	vector <int> &ali1,vector <int> &ali2)
{
	int i;
	int size=(int)AFP_Cor.size();
	int fix_or_not=0;
	//tail fix
	for(i=0;i<size-1;i++)
	{
		//cur tail
		int len1=AFP_Cor[i][2];
		int tail1=AFP_Cor[i][0]+len1-1;
		int tail2=AFP_Cor[i][1]+len1-1;
		//next head
		int len2=AFP_Cor[i+1][2];
		int head1=AFP_Cor[i+1][0];
		int head2=AFP_Cor[i+1][1];
		//check continue
		if( (str1[tail1]==str2[tail2]) && (str1[head1]==str2[head2]) )
		{
			continue;
		}
		if( (str1[tail1]!=str2[tail2]) && (str1[head1]!=str2[head2]) )
		{
			continue;
		}
		//check move
		if( (str1[tail1]==str2[tail2]) && (str1[head1]!=str2[head2]) )
		{
			if(str1[tail1+1]==str2[tail2+1])
			{
				ali1[head1]=-1;
				ali2[head2]=-1;
				ali1[tail1+1]=tail2+1;
				ali2[tail2+1]=tail1+1;
				//record
				fix_or_not=1;
			}
		}
		if( (str1[tail1]!=str2[tail2]) && (str1[head1]==str2[head2]) )
		{
			if(str1[head1-1]==str2[head2-1])
			{
				ali1[tail1]=-1;
				ali2[tail2]=-1;
				ali1[head1-1]=head2-1;
				ali2[head2-1]=head1-1;
				//record
				fix_or_not=1;
			}
		}
	}
	//return
	return fix_or_not;
}

//============= Output Alignment ========//
//-> fasta_output_simp
void FASTA_Output_Simp(FILE *fp,string &nam1,string &nam2,
	const char *ami1,const char *ami2,
	vector<pair<int,int> > &alignment,
	string &assess)
{
	//alignment->output
	//output
	char c;
	int i;
	int ii,jj;
	int size=(int)alignment.size();
	//--> output seq1
	fprintf(fp,">%s\n",nam1.c_str());
	for(i=0;i<size;i++)
	{
		ii=alignment[i].first;
		if(ii<=0)fprintf(fp,"-");
		else
		{
			c=ami1[ii-1];
			fprintf(fp,"%c",c);
		}
	}
	fprintf(fp,"\n");
	//--> output assess
	if(assess!="")
	{
		fprintf(fp,"%s\n",assess.c_str());
	}
	//--> output seq2
	fprintf(fp,">%s\n",nam2.c_str());
	for(i=0;i<size;i++)
	{
		jj=alignment[i].second;
		if(jj<=0)fprintf(fp,"-");
		else
		{
			c=ami2[jj-1];
			fprintf(fp,"%c",c);
		}
	}
	fprintf(fp,"\n");
}

//---------- main ----------//
int main(int argc,char **argv)
{
	//---- nucleotide sequence dynaprog ----//
	{
		if(argc<4)
		{
			printf("DNA_DynaProg <fasta1> <fasta2> <ali_out> \n");
			exit(-1);
		}
		string fasta1=argv[1];
		string fasta2=argv[2];
		string ali_out=argv[3];
		int retv;

		//-> 1. read fasta sequence
		int skip=1;
		string seqres1;
		retv=Read_FASTA_SEQRES(fasta1,seqres1,skip);
		if(retv<=0)
		{
			fprintf(stderr,"fasta1 %s error \n",fasta1.c_str());
			exit(-1);
		}
		string seqres2;
		retv=Read_FASTA_SEQRES(fasta2,seqres2,skip);
		if(retv<=0)
		{
			fprintf(stderr,"fasta2 %s error \n",fasta2.c_str());
			exit(-1);
		}

		//-> 2. dynamic programming alignment
		vector<pair<int,int> > WWW_alignment;
		int match;
		double sco=process_oriami_record_simp(seqres1.c_str(),seqres2.c_str(),WWW_alignment,match);

		//-> 3. fix alignment
		vector <int> ali1;
		vector <int> ali2;
		AliPair_To_Ali1_Ali2(seqres1.length(),seqres2.length(),
			WWW_alignment,ali1,ali2);
		for(int i=0;i<5;i++)
		{
			vector <vector <int> > AFP_Cor;
			Ali_To_Cor(ali2,AFP_Cor);
			int retv=Head_Tail_Fix(seqres1,seqres2,AFP_Cor,ali1,ali2);
			if(retv==0)break;
		}
		Ali1_To_AliPair(seqres1.length(),seqres2.length(),
			ali1,WWW_alignment);

		//-- remove head and tail gap --//
		int iden=0;
		int igap=0;
		int ilen=0;
		string assess="";
		{
			//--> determine start
			int start=-1;
			for(int i=0;i<(int)WWW_alignment.size();i++)
			{
				int ii=WWW_alignment[i].first;
				int jj=WWW_alignment[i].second;
				if(ii>0 && jj>0)
				{
					start=i;
					break;
				}
			}
			//--> determine end
			int end=-1;
			for(int i=(int)WWW_alignment.size()-1;i>=0;i--)
			{
				int ii=WWW_alignment[i].first;
				int jj=WWW_alignment[i].second;
				if(ii>0 && jj>0)
				{
					end=i;
					break;
				}
			}
			//--> final check
			if(start==-1 || end==-1)
			{
				fprintf(stderr,"BAD HERE !!! Null Position Aligned !! \n");
				exit(-1);
			}
			//--> determine
			for(int i=0;i<start;i++)assess.push_back('.');
			for(int i=start;i<=end;i++)
			{
				int ii=WWW_alignment[i].first;
				int jj=WWW_alignment[i].second;
				if(ii>0 && jj>0)
				{
					if(seqres1[ii-1]==seqres2[jj-1])
					{
						assess.push_back('|');
						iden++;
					}
					else
					{
						assess.push_back('X');
					}
				}
				else
				{
					assess.push_back('-');
					igap++;
				}
				ilen++;
			}
			for(int i=end+1;i<(int)WWW_alignment.size();i++)assess.push_back('.');
		}

		//-> 3. output alignment
		string nam1;
		getBaseName(fasta1,nam1,'/','.');
		string nam2;
		getBaseName(fasta2,nam2,'/','.');
		FILE *fp=fopen(ali_out.c_str(),"wb");
		fprintf(fp,"%s %s -> %lf -> %d %d %d -> %lf %lf -> %d/%d(%lf) | %d/%d(%lf)\n",
			nam1.c_str(),nam2.c_str(),sco,
			iden,seqres1.length(),seqres2.length(),
			1.0*iden/seqres1.length(),1.0*iden/seqres2.length(),
			iden,ilen,1.0*iden/ilen,igap,ilen,1.0*igap/ilen);
		FASTA_Output_Simp(fp,nam1,nam2,seqres1.c_str(),seqres2.c_str(),WWW_alignment,assess);
		printf("%s %s -> %lf -> %d %d %d -> %lf %lf -> %d/%d(%lf) | %d/%d(%lf)\n",
			nam1.c_str(),nam2.c_str(),sco,
			iden,seqres1.length(),seqres2.length(),
			1.0*iden/seqres1.length(),1.0*iden/seqres2.length(),
			iden,ilen,1.0*iden/ilen,igap,ilen,1.0*igap/ilen);
		fclose(fp);
		
		//exit
		exit(0);
	}
}
