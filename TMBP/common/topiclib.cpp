/*---------------------------------------------------
* file:    topiclib.c
* purpose: routines for topic modeling
* version: 1.0
* author:  j.zeng@ieee.org
* date:    2012-02-09
*-------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

typedef unsigned long uint32;

#define SQR(a)   ((a) * (a))

#define irand()  randomMT()
#define drand() (randomMT()/4294967296.0)

void seedMT(uint32 seed);
uint32 randomMT(void);

double digamma(double x);
int *ivec(int n);
double *dvec(int n);
void updatesort(int n, int *x, int *indx, int *revindx, int ip, int im);
void insertionsort(int n, double *x, int *indx);
void isort(int n, int *x, int direction, int *indx);
void dsort(int n, double *x, int direction, int *indx);
double etime();

static int icomp(const void *, const void *); /* comparison for isort */
static int dcomp(const void *, const void *); /* comparison for dsort */
static int    *icomp_vec;                     /*  data used for isort */
static double *dcomp_vec;                     /*  data used for dsort */

/*------------------------------------------
* allocation routines
* ivec
* dvec
*------------------------------------------ */

double *dvec(int n) //
{
  double *x = (double *) mxCalloc(n, sizeof(double));
  return x;
}

int *ivec(int n) //
{
  int *x = (int *) mxCalloc(n, sizeof(int));
  return x;
}

/*------------------------------------------
* sort: call qsort library function
* isort
* dsort
* insertionsort
* updatesort
*------------------------------------------ */

void updatesort(int n, int *x, int *indx, int *revindx, int ip, int im) //
{
	int tmp;

	// INCREMENT
	// did ++ get bigger than prev?
	ip = revindx[ip];
	while (ip>0 && x[indx[ip]] > x[indx[ip-1]]) {
		// swap indx
		tmp        = indx[ip];
		indx[ip]   = indx[ip-1];
		indx[ip-1] = tmp;
		// swap revindx
		tmp                 = revindx[indx[ip]];
		revindx[indx[ip]]   = revindx[indx[ip-1]];
		revindx[indx[ip-1]] = tmp;
		ip--;
	}

	// DECREMENT
	// did -- get smaller than next?
	im = revindx[im];
	while (im<n-1 && x[indx[im]] < x[indx[im+1]]) {
		// swap indx
		tmp        = indx[im];
		indx[im]   = indx[im+1];
		indx[im+1] = tmp;
		// swap revindx
		tmp                 = revindx[indx[im]];
		revindx[indx[im]]   = revindx[indx[im+1]];
		revindx[indx[im+1]] = tmp;
		im++;
	}
}


void insertionsort(int n, double *x, int *indx) // descending order
{
	int tmp, i, k;

	for (i=1; i<n; i++) {
		for (k=i; k>0 && x[indx[k]]>x[indx[k-1]]; k--) {
			tmp = indx[k];
			indx[k] = indx[k-1];
			indx[k-1] = tmp;
		}
	}
}

void isort(int n, int *x, int direction, int *indx) // +1 ascending order -1 descending order
{
	int i;
	icomp_vec = ivec(n);
	for (i=0; i<n; i++) {
		icomp_vec[i] = direction*x[i];
		indx[i] = i;
	}
	qsort(indx, n, sizeof(int), icomp);
	mxFree(icomp_vec);
}

static int icomp(const void *pl, const void *p2)
{
	int i = * (int *) pl;
	int j = * (int *) p2;
	return (icomp_vec[i] - icomp_vec[j]);
}

void dsort(int n, double *x, int direction, int *indx) // +1 ascending order -1 descending order
{
	int i;
	dcomp_vec = dvec(n);
	for (i=0; i<n; i++) {
		dcomp_vec[i] = direction*x[i];
		indx[i] = i;
	}
	qsort(indx, n, sizeof(int), dcomp);
	mxFree(dcomp_vec);
}

static int dcomp(const void *p1, const void *p2)
{
	int i = * (int *) p1;
	int j = * (int *) p2;
	return dcomp_vec[i] > dcomp_vec[j] ? 1:-1;
}

/* CPU time */
double etime() //
{
	static double last_clock = 0;
	static double now_time = 0;
	last_clock = now_time;
	now_time = (double) clock ();
	return (double) (now_time - last_clock) / CLOCKS_PER_SEC;
}

/* digamma funciton implemented by David Blei */
double digamma(double x)
{
	double p;
	x = x + 6;
	p = 1/(x*x);
	p = (((0.004166666666667*p-0.003968253986254)*p+
		0.008333333333333)*p-0.083333333333333)*p;
	p = p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	return p;
}

typedef unsigned long uint32;

#define N              (624)                 // length of state vector
#define M              (397)                 // a period parameter
#define K              (0x9908B0DFU)         // a magic constant
#define hiBit(u)       ((u) & 0x80000000U)   // mask all but highest   bit of u
#define loBit(u)       ((u) & 0x00000001U)   // mask all but lowest    bit of u
#define loBits(u)      ((u) & 0x7FFFFFFFU)   // mask     the highest   bit of u
#define mixBits(u, v)  (hiBit(u)|loBits(v))  // move hi bit of u to hi bit of v

static uint32   state[N+1];     // state vector + 1 extra to not violate ANSI C
static uint32   *next;          // next random value is computed from here
static int      left = -1;      // can *next++ this many times before reloading


void seedMT(uint32 seed)
 {
    //
    // We initialize state[0..(N-1)] via the generator
    //
    //   x_new = (69069 * x_old) mod 2^32
    //
    // from Line 15 of Table 1, p. 106, Sec. 3.3.4 of Knuth's
    // _The Art of Computer Programming_, Volume 2, 3rd ed.
    //
    // Notes (SJC): I do not know what the initial state requirements
    // of the Mersenne Twister are, but it seems this seeding generator
    // could be better.  It achieves the maximum period for its modulus
    // (2^30) iff x_initial is odd (p. 20-21, Sec. 3.2.1.2, Knuth); if
    // x_initial can be even, you have sequences like 0, 0, 0, ...;
    // 2^31, 2^31, 2^31, ...; 2^30, 2^30, 2^30, ...; 2^29, 2^29 + 2^31,
    // 2^29, 2^29 + 2^31, ..., etc. so I force seed to be odd below.
    //
    // Even if x_initial is odd, if x_initial is 1 mod 4 then
    //
    //   the          lowest bit of x is always 1,
    //   the  next-to-lowest bit of x is always 0,
    //   the 2nd-from-lowest bit of x alternates      ... 0 1 0 1 0 1 0 1 ... ,
    //   the 3rd-from-lowest bit of x 4-cycles        ... 0 1 1 0 0 1 1 0 ... ,
    //   the 4th-from-lowest bit of x has the 8-cycle ... 0 0 0 1 1 1 1 0 ... ,
    //    ...
    //
    // and if x_initial is 3 mod 4 then
    //
    //   the          lowest bit of x is always 1,
    //   the  next-to-lowest bit of x is always 1,
    //   the 2nd-from-lowest bit of x alternates      ... 0 1 0 1 0 1 0 1 ... ,
    //   the 3rd-from-lowest bit of x 4-cycles        ... 0 0 1 1 0 0 1 1 ... ,
    //   the 4th-from-lowest bit of x has the 8-cycle ... 0 0 1 1 1 1 0 0 ... ,
    //    ...
    //
    // The generator's potency (min. s>=0 with (69069-1)^s = 0 mod 2^32) is
    // 16, which seems to be alright by p. 25, Sec. 3.2.1.3 of Knuth.  It
    // also does well in the dimension 2..5 spectral tests, but it could be
    // better in dimension 6 (Line 15, Table 1, p. 106, Sec. 3.3.4, Knuth).
    //
    // Note that the random number user does not see the values generated
    // here directly since reloadMT() will always munge them first, so maybe
    // none of all of this matters.  In fact, the seed values made here could
    // even be extra-special desirable if the Mersenne Twister theory says
    // so-- that's why the only change I made is to restrict to odd seeds.
    //

    register uint32 x = (seed | 1U) & 0xFFFFFFFFU, *s = state;
    register int    j;

    for(left=0, *s++=x, j=N; --j;
        *s++ = (x*=69069U) & 0xFFFFFFFFU);
 }


uint32 reloadMT(void)
 {
    register uint32 *p0=state, *p2=state+2, *pM=state+M, s0, s1;
    register int    j;

    if(left < -1)
        seedMT(4357U);

    left=N-1, next=state+1;

    for(s0=state[0], s1=state[1], j=N-M+1; --j; s0=s1, s1=*p2++)
        *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);

    for(pM=state, j=M; --j; s0=s1, s1=*p2++)
        *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);

    s1=state[0], *p0 = *pM ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
    s1 ^= (s1 >> 11);
    s1 ^= (s1 <<  7) & 0x9D2C5680U;
    s1 ^= (s1 << 15) & 0xEFC60000U;
    return(s1 ^ (s1 >> 18));
 }


uint32 randomMT(void)
 {
    uint32 y;

    if(--left < 0)
        return(reloadMT());

    y  = *next++;
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);
    return(y);
 }
