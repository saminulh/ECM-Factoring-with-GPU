#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <cuda_runtime.h>

#define LIMIT 1200
#define NEGATOR 4294967295
#define SIZE 4
#define ul unsigned
#define ull unsigned long long

typedef struct{
  ul* x;
  ul* y;
}Point;

//Compare two numbers
__device__ int compare(ul* f, ul* s){
  int i;
  for (i = SIZE -1; i >= 0; i--){
    if (f[i] < s[i]) return -1;
    if (f[i] > s[i]) return 1;
  }
  return 0;
}

//Add two numbers
__device__ void add(ul* a, ul* b, ul* d){
  int carry = 0, i;
  ull temp;
  for (i = 0; i < SIZE; i++){
    temp = (ull) a[i] + (ull) b[i] + (ull) carry;
    d[i] = temp & (ull) NEGATOR;
    carry = temp >> 32;
  }
  
}

//Subtract two numbers
__device__ void subtract(ul* s, ul* n, ul* d){
  ull temp;
  int takeaway = 0, i;
  for (i = 0; i < SIZE; i++){
    if (s[i] < n[i] || s[i] < n[i] + takeaway || takeaway*n[i] == NEGATOR){
      temp = ((ull) 1 << 32) + (ull) s[i] - (ull) takeaway - (ull) n[i];
      takeaway = 1;
    }
    else{
      temp = s[i] - n[i] - takeaway;
      takeaway = 0;
    }
    d[i] = (ul) temp;
  }
}

//Distribute 64-bit number between two 32-bit numbers
__device__ void assignSandC(ull temp, ul *S, ul *C){
  *S = temp;
  *C = temp >> 32;
}

//Montgomery CIOS multiplication
__device__ void montMul(ul* a, ul* b, ul *N, ul nInv, ul *d){

  int i,j;
  ul C,S, m, overflow = 0, overflow2 = 0;
  ull temp;
  
  //Clear destination
  for (i = 0; i < SIZE; i++){
    d[i] = 0;
  }
  
  for (i = 0; i < SIZE; i++){
    C = 0;

    //Multiplication Step
    for (j = 0; j < SIZE; j++){
      assignSandC((ull) d[j] + (ull) a[j]*(ull) b[i] + (ull) C, &S, &C);
      d[j] = S;
    }
    
    assignSandC( (ull) overflow + (ull) C, &S, &C);
    overflow = S; overflow2 = C;
    C = 0;
    
    m = (ul) (d[0]*nInv)%((ull) 1<<32);
    assignSandC( (ull) d[0] + (ull) m * (ull)N[0], &S, &C);
    
    //Reduction Step
    for (j=1; j<SIZE; j++){
      assignSandC((ull) d[j] + (ull) m * (ull) N[j] + (ull) C, &S, &C);
      d[j-1] = S;
    }
    assignSandC( (ull) overflow + (ull) C, &S, &C);
    d[SIZE-1] = S; overflow = overflow2 + C;
  }
}

//Standard Right-to-Left binary exponentiation
__device__ void montExp(ul *b, ul e, ul *N, ul nInv, ul *d){
  int i, dswaps = 0;
  bool first = true;
  
  ul tmpArr[SIZE], baseArr[SIZE], baseTmpArr[SIZE];
  ul *tmp = tmpArr, *tmp2, *base = baseArr, *baseTmp = baseTmpArr, *baseTmp2;
  
  memcpy(base, b, SIZE*sizeof(ul));
  
  for (i = 0; i < SIZE; i++){
    d[i] = 0;
  }
  
  while (e > 0){
    if (e & 1){
      if (first){
        memcpy(tmp, base, SIZE*sizeof(ul));
        first = false;
      }
      else{
        montMul(d, base, N, nInv, tmp);
      }
      tmp2 = d; d = tmp; tmp = tmp2;
      dswaps = 1-dswaps;
    }
    e >>= 1;
    montMul(base, base, N, nInv, baseTmp);
    baseTmp2 = base; base = baseTmp; baseTmp = baseTmp2;
  }
  if (dswaps){
    memcpy(tmp, d, SIZE*sizeof(ul));
  }
}

//Add two Montgomery Curves
__device__ void addPoints(Point p1, Point p2, Point diff, ul *N, ul nInv, Point d, ul *ONE, ul *ZERO){
  
  if (compare(ZERO, p1.x) == 0 && compare(ZERO, p1.y) == 0){
    memcpy(d.x, p2.x, SIZE*sizeof(ul));
    memcpy(d.y, p2.y, SIZE*sizeof(ul));
    return;
  }
  if (compare(ZERO, p2.x) == 0 && compare(ZERO, p2.y) == 0){
    memcpy(d.x, p1.x, SIZE*sizeof(ul));
    memcpy(d.y, p1.y, SIZE*sizeof(ul));
    return;
  }
  int i;
  
  //UNeg and VNeg represents if the variables U and V actually contain -U and -V of the formulaes
  bool UNeg = compare(p1.x, p1.y) == -1, VNeg = compare(p2.x, p2.y) == -1;
  
  ul U[SIZE], V[SIZE], tmp[SIZE], tmp2[SIZE];
  
  if (UNeg) subtract(p1.y, p1.x, tmp);
  else subtract(p1.x, p1.y, tmp);
  add(p2.x, p2.y, tmp2);
  montMul(tmp, tmp2, N, nInv, U);
  
  if (VNeg) subtract(p2.y, p2.x, tmp);
  else subtract(p2.x, p2.y, tmp);
  add(p1.x, p1.y, tmp2);
  montMul(tmp, tmp2, N, nInv, V);
  
  if (UNeg ^ VNeg){
    if (compare(U,V) == 1) subtract(U,V,tmp);
    else subtract(V,U,tmp);
  }
  else add(U,V,tmp);
  montMul(tmp, tmp, N, nInv, tmp2);
  montMul(diff.y, tmp2, N, nInv, d.x);
  
  if (UNeg ^ VNeg) add(U,V,tmp);
  else {
    if (compare(U,V) == 1) subtract(U,V,tmp);
    else subtract(V,U,tmp);
  }
  montMul(tmp, tmp, N, nInv, tmp2);
  montMul(diff.x, tmp2, N, nInv, d.y);
}

//Double two points on a Montgomery curve
__device__ void doublePoint(Point p, ul *S, ul *N, ul nInv, Point d){
  ul dp[SIZE], dm[SIZE], T[SIZE], tmp[SIZE], tmp2[SIZE];
  
  add(p.x, p.y, tmp);
  
  montMul(tmp, tmp, N, nInv, dp);
  
  
  if (compare(p.x, p.y) == 1) subtract(p.x, p.y, tmp);
  else subtract(p.y, p.x, tmp);
  montMul(tmp, tmp, N, nInv, dm);
  
  if (compare(dp, dm) > -1) subtract(dp, dm, T);
  else{
    subtract(dm, dp, tmp);
    subtract(N, tmp, T);
  }
  
  montMul(dp, dm, N, nInv, d.x);
  
  montMul(S, T, N, nInv, tmp);
  add(tmp, dm, tmp2);
  montMul(T, tmp2, N, nInv, d.y);
}

//Left-to-right binary scaling with Doubles and Adds
//Keeps track of nP and (n-1)P so that the difference is always P
__device__ void scalePoint(Point p, ul c, ul *S, ul *N, ul nInv, Point d, ul *ONE, ul *ZERO){
  ul temp;
  int i;
  bool started = false;
  //if (threadIdx.x + blockIdx.x == 0) printf("%u\n", c);
  ul t[SIZE], t2[SIZE], t3[SIZE], t4[SIZE], t5[SIZE], t6[SIZE], t7[SIZE], t8[SIZE];
    
  for (i = 0; i < SIZE; i++){
    t[i] = 0;
    t2[i] = 0;
    t3[i] = p.x[i];
    t4[i] = p.y[i];
  }
  
  Point kp = {t, t2}, kp1p = {t3, t4}, twokp1 = {t5, t6}, doub =  {t7, t8}, pTemp;

  for (temp = (ul) 1<<31; temp > 0; temp >>= 1){
    if (!started){
      if ((c&temp) != 0){
        started = true;
      }
    }
    if(started){
      addPoints(kp, kp1p, p, N, nInv, twokp1, ONE, ZERO);
      
      if ((c&temp) == 0) {
        doublePoint(kp, S, N, nInv, doub);
        pTemp = kp; kp = doub; doub = pTemp;
        pTemp = kp1p; kp1p = twokp1; twokp1 = pTemp;
      }
      else{
        doublePoint(kp1p, S, N, nInv, doub);
        pTemp = kp; kp = twokp1; twokp1 = pTemp;
        pTemp = kp1p; kp1p = doub; doub = pTemp;
      }
    }
  }
  
  memcpy(d.x, kp.x, SIZE*sizeof(ul));
  memcpy(d.y, kp.y, SIZE*sizeof(ul));
}

//Convert s to Montgomery Representation by shifting right SIZE*32 times
__device__ void toMont(ul *s, ul *N, ul *d){
  int i, j, carry, tmpCarry, swaps = 0;
  
  ul tempArr[SIZE], *temp, *memTemp;
  temp = tempArr;  
  
  memcpy(d, s, SIZE*sizeof(ul));
  
  for (i = 0; i < SIZE*32; i++){
    carry = 0;
    for (j = 0; j < SIZE; j++){
      tmpCarry = (d[j]>>31) != 0;
      d[j] = ((ul) (d[j] << 1)) + carry;
      carry = tmpCarry;
    }
    
    if (compare(d, N) != -1){
      
      subtract(d, N, temp);
      memTemp = d; d = temp; temp = memTemp;
      swaps = 1-swaps;
    }
  }
  
  if (swaps == 1){
    memcpy(temp, d, SIZE*sizeof(ul));
  }
}

//Integer division
__device__ void idiv(ul *b, ul *a, ul *quotient, ul *mod){
  int i, shifts, carry, tmpCarry;
  ul tempA[SIZE];
  
  memcpy(mod, b, SIZE*sizeof(ul));
  for (i = 0; i < SIZE; i++){
    quotient[i] = 0;
  }
  while (compare(a, mod) != 1){
    shifts = 0;
    memcpy(tempA, a, SIZE*sizeof(ul));
    
    while (compare(tempA, mod) != 1){
      carry = 0;
      for (i = 0; i < SIZE; i++){
        tmpCarry = (tempA[i]>>31) != 0;
        tempA[i] = ((ul) (tempA[i] << 1)) + carry;
        carry = tmpCarry;
      }
      shifts += 1;
    }
    for (i = 0; i < SIZE; i++){
      carry = 0;
      if (i < SIZE-1) carry = tempA[i+1]%2;
      tempA[i] = (tempA[i] >> 1) + (carry << 31);
    }
    shifts -= 1;
    
    subtract(mod, tempA, mod);
    quotient[shifts/32] += 1<<(shifts%32);
  }
  
}

//Standard multiplication
__device__ void regMul(ul* a, ul* b, ul *d){
  int i,j;
  ul C,S, overflow = 0, overflow2 = 0;
  
  for (i = 0; i < SIZE; i++){
    d[i] = 0;
  }
  
  for (i = 0; i < SIZE; i++){
    C = 0;
    for (j = 0; j < SIZE; j++){
      assignSandC((ull) d[j+i] + (ull) a[j]*(ull) b[i] + (ull) C, &S, &C);
      d[j+i] = S;
    }
    
  }
}

//Extended Euclidean algorithm for GCD and modular inverse
__device__ void ee(ul *a, ul *b, ul* ainv, ul* gcd, ul *ZERO){
  
  int i;
    
  ul sArr[SIZE], old_sArr[SIZE], rArr[SIZE], old_rArr[SIZE], quotientArr[SIZE], prodArr[SIZE], tempArr[SIZE], tTemp[SIZE];
  
  ul *s = sArr, *old_s = old_sArr, *r = rArr, *old_r = old_rArr, *quotient = quotientArr, *prod = prodArr, *temp = tempArr, *temp2;
    
  for (i = 0; i < SIZE; i++){
    s[i] = 0;
    old_s[i] = i == 0;
    r[i] = b[i];
    old_r[i] = a[i];
  }
  while (compare(r,ZERO) != 0){
    
    idiv(old_r, r, quotient, temp);
    temp2 = old_r;
    old_r = r;
    r = temp;
    temp = temp2;
    
    regMul(quotient, s, prod);
    temp2 = temp;
    temp = old_s;
    old_s = s;
    if (compare(temp, prod) != -1){
    	subtract(temp,prod,temp);
	 }
	 else{
	 	subtract(prod, temp, temp);
	 	idiv(temp, b, quotient, tTemp);
	 	subtract(b, tTemp, temp);
	 }
    s = temp;
    temp = temp2;
  }
  memcpy(ainv, old_s, SIZE*sizeof(ul));
  memcpy(gcd, old_r, SIZE*sizeof(ul));
}

//Thread that handles all the curve computations
__global__ void curveThread(ul *N, ul *Rsquare, ul *nI, ul *primes, ul *ONE, ul *ZERO, int disp){

	//Generate paramter for curve operations
	ul sigma = 6+(blockIdx.x*blockDim.x)+threadIdx.x+disp;
	ul tempId = threadIdx.x;
	int i, j;
	
	ul S[SIZE], t[SIZE], t2[SIZE], t3[SIZE], t4[SIZE], t5[SIZE], t6[SIZE], x[SIZE], z[SIZE], U[SIZE], V[SIZE], SMALLNUM[SIZE], nInv = *nI;

	//Store primes in shared memory
	__shared__ extern ul p[];
	while(tempId < LIMIT){
		p[tempId] = primes[tempId];
		tempId += blockDim.x;
	}
	__syncthreads();
	

	for (i = 0; i < SIZE; i++){
	  t[i] = 0;
     t2[i] = 0;
     t3[i] = (i==0)*sigma;
     t4[i] = 0;
     t5[i] = 0;
     t6[i] = 0;
     U[i] = 0;
     V[i] = 0;
     SMALLNUM[i] = 0;
   }

   //Generate Curve and point as per Suyama's Parametrization

	   SMALLNUM[0] = 5;
	   regMul(t3, t3, t);
	   subtract(t, SMALLNUM, t);
	   SMALLNUM[0] = 4;
	   regMul(t3, SMALLNUM, t2);
	   t3[0] = 0;
   
   
   
   montMul(t, Rsquare, N, nInv, U);
   montMul(t2, Rsquare, N, nInv, V);
   
   montExp(U, 3, N, nInv, x);
   montExp(V, 3, N, nInv, z);
   
   t2[0] *= 16;

   montMul(t2, Rsquare, N, nInv, t);
	   
   montMul(t, x, N, nInv, t3);
	   
   montMul(t3, ONE, N, nInv, t5);

   ee(t5, N, t, t2, ZERO);
   montMul(t, Rsquare, N, nInv, t2);
   
   if(compare(U,V) == 1){
     subtract(U, V, t);
     montMul(t, t, N, nInv, t3);
     montMul(t, t3, N, nInv, t4);
     subtract(N, t4, t4);
   }
   else{
     subtract(V, U, t);
     montMul(t, t, N, nInv, t3);
     montMul(t, t3, N, nInv, t4);
   }
   
   SMALLNUM[0] = 3;
   montMul(SMALLNUM, Rsquare, N, nInv, t3);
   montMul(U, t3, N, nInv, t5);
   add(t5, V, t5);
   
   montMul(t5, t4, N, nInv, t6);
   montMul(t6, t2, N, nInv, S);
   
   Point P = {x, z}, D = {t3, t4}, twoP = {t5, t6}, pTemp;
		   

   //Scale by k
   for (i = 0; i < LIMIT; i+=2){
     scalePoint(P, p[i]*p[i+1], S, N, nInv, D, ONE, ZERO);
     pTemp = D; D = P; P = pTemp;
   }
   
   ee(P.x, N, t5, t6, ZERO);
   
	if (compare(t6, ONE) != 0 && compare(t6,N) != 0) 
   printf("%i: %u %u %u %u\n", sigma, t6[0], t6[1], t6[2], t6[3]);

}

//Setup preliminary information (includes the input N)
__global__ void setup(ul *N, ul *primes, ul *nInv, ul *Rsquare, ul *ONE, ul *ZERO){
  ul B1 = 20 * LIMIT, numFound = 0;
  
  int i,j;
  if (B1 < 100) B1 = 100;
  
  int *primeOrNot;
  size_t size = B1*sizeof(int);
  primeOrNot = (int*) malloc(B1*sizeof(int));
  for (i = 0; i < B1; i++){
     primeOrNot[i] = 0;  
  }
    
  ul t[SIZE], t2[SIZE], t3[SIZE];

  i = 0;  
  while (i < B1 && numFound < LIMIT){
    if (i > 1 && !primeOrNot[i]){
    	int temp = 1;
    	while (temp < B1) temp *= i;
    	temp = temp/i;
      primes[numFound] = temp;
      numFound += 1;
      for (j = 2*i; j < B1; j += i){
        primeOrNot[j] = 1;
      }
    }
    i += 1;
  }
  free(primeOrNot);
  
  for (i = 0; i < SIZE; i++){
    ONE[i] = i==0; 
    ZERO[i] = 0;
    t[i] = i==1;
  }
  
  //Input the value of N here

  N[0] = 3521154867; N[1] = 785645112; N[2] = 2260700419; N[3] = 0; //N[4] = 0; N[5] = 0;
  //N[0] = 3619132259; N[1] = 3251792858; N[2] = 2426605881; N[3] = 457503228; N[4] = 1;// N[5] = 0;
    

  //Find negative of modular inverse of N
    
  ee(N, t, t2, t3, ZERO);
 
  *nInv = NEGATOR-t2[0]+1;

  //Find conversion number R^2
  toMont(ONE, N, t);
  toMont(t, N, Rsquare);
}

int main(){

  
  ul *N, *primes, B1, *S, *t, *t2, *t3, *t4, *t5, *t6, *Rsquare, *x, *z, *U, *V, *ONE, *ZERO, *SMALLNUM, *nInv;
  
  ul *hN;
  
  size_t size = sizeof(ul)*LIMIT;
  cudaMalloc((void**) &primes, size);
  size = sizeof(ul)*SIZE;
  cudaMalloc((void**) &N,size);
  cudaMalloc((void**) &Rsquare, size);
  cudaMalloc((void**) &SMALLNUM, size);
  cudaMalloc((void**) &ZERO, size);
  cudaMalloc((void**) &ONE, size);
  size = sizeof(ul);
  cudaMalloc((void**) &nInv, size);
  
  
  hN = (ul*) malloc(sizeof(ul)*SIZE);
  
  setup<<<1,1>>>(N, primes, nInv, Rsquare, ONE, ZERO);

  //Set number of threads here
  curveThread<<<1024, 128, LIMIT*sizeof(ul)>>>(N, Rsquare, nInv, primes, ONE, ZERO, 0);
  
  //cudaMemcpy(hN, N, sizeof(ul)*SIZE, cudaMemcpyDeviceToHost);
  //printf("%u\n", hN[0]);
  printf("DONE\n");
    
  return 0;
}