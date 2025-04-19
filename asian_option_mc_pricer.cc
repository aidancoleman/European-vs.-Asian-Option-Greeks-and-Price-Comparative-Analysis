#include <iostream>
#include <cmath>
#include <algorithm> //for std::max operation
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h> //for gsl_ran_gaussian
#include <iomanip> //for std::setprecision
class Asian{
public:
    /* ASIAN OPTION MONTE CARLO PRICER
     n = number of Monte Carlo Samples
     S0 = initial spot price today
     K = strike price
     sigma = annual volatility
     r = risk free interest rate (annual)
     d = days to expiry
     */
    double price(const long long n, const double S0, const double K, const int d, const double sigma, const double r){
        double dt = 1.0/255.0; //change (daily time increment in years)
        double T = d*dt; //time to expiry in years (for discounted price)
        gsl_rng_env_setup(); //setup environment for rng
        gsl_rng *rng = gsl_rng_alloc(gsl_rng_default); //default gsl rng = Tausworthe
        //Outer loop for each Monte Carlo Sample
        double sampleAvg = 0.0;
        double totalPayoff = 0.0;
        for(long long i = 0; i<n; i++){
            double ST = S0; //initialise price at expiry as starting underlying spot price, then take product of prices over consecutive days until expiry
            double payoffSum = S0;//initialise sum of the payoffs, we divide this by the number of days at the end to get the average asset price over the time period
            //Inner loop for tracking the asset price over the days until expiry
            for(int k = 1; k < d; k++){
                double z = gsl_ran_gaussian(rng, 1.0); //generate random standard normal variable with gsl_ran_gaussian(rng, 1)
                ST *= std::exp((r - 0.5*(sigma)*(sigma))*(dt) + sigma*(std::sqrt(dt))*(z));
                payoffSum += ST;
            }
            double pathAvg = payoffSum/d;
            double payoff = std::max(pathAvg - K, 0.0);
            totalPayoff += payoff;
        }
        gsl_rng_free(rng);
        sampleAvg = totalPayoff / n;
        return sampleAvg*std::exp(-r*T); //discounted average asset price
    }
    /*Calculate Delta using bump and revalue (central difference)
     */
    double delta(const long long n, const double S0, const double K, const int d, const double sigma, const double r){
        double h = S0*1e-6; //sufficiently small h = 0.000001 x S0
        const unsigned long seed = 12345578; //same seed for prng for both cases where price increases and price decreases for consistency, randomness (noise) not accounting for difference in results, same sequence of random numbers generated in both runs
        /*
         WHEN PRICE INCREASES (S0 + h)
         */
        gsl_rng_env_setup();
        gsl_rng *rng_up = gsl_rng_alloc(gsl_rng_default); //default gsl rng = Tausworthe
        gsl_rng_set(rng_up, seed);
        double pi_up = price(n, S0 + h, K, d, sigma, r, rng_up); //call overloaded price function so we can pass rng to it and use the same seed for both spot price fluctuations for consistency, both cases use the exact same random paths in both runs (up and down in price)
        gsl_rng_free(rng_up);
        /*
         WHEN PRICE DECREASES (S0 - h)
         */
        gsl_rng_env_setup();
        gsl_rng *rng_down = gsl_rng_alloc(gsl_rng_default); //default gsl rng = Tausworthe
        gsl_rng_set(rng_down, seed);
        double pi_down = price(n, S0 - h, K, d, sigma, r, rng_down);
        gsl_rng_free(rng_down);
        /*
         CALCULATE SENSITIVITY
         */
        double delta = (pi_up - pi_down)/(2.0*h);
        return delta;
    }
    /*
     Calculate Vega with central difference bump and revalue
     */
    double vega(const long long n, const double S0, const double K, const int d, const double sigma, const double r){
        double h = 1e-3; //sufficiently small h = 0.001
        const unsigned long seed = 12345578; //same seed for prng for both cases where sigma increases and sigma decreases for consistency, randomness (noise) not accounting for difference in results, same sequence of random numbers generated in both runs
        /*
         WHEN SIGMA INCREASES (sigma + h)
         */
        gsl_rng_env_setup();
        gsl_rng *rng_up = gsl_rng_alloc(gsl_rng_default); //default gsl rng = Tausworthe
        gsl_rng_set(rng_up, seed);
        double pi_up = price(n, S0, K, d, sigma+h, r, rng_up); //call overloaded price function so we can pass rng to it and use the same seed for both volatility fluctuations for consistency, both cases use the exact same random paths in both runs (up and down in volatility)
        gsl_rng_free(rng_up);
        /*
         WHEN SIGMA DECREASES (sigma - h)
         */
        gsl_rng_env_setup();
        gsl_rng *rng_down = gsl_rng_alloc(gsl_rng_default); //default gsl rng = Tausworthe
        gsl_rng_set(rng_down, seed);
        double pi_down = price(n, S0, K, d, sigma-h, r, rng_down);
        gsl_rng_free(rng_down);
        /*
         CALCULATE SENSITIVITY
         */
        double vega = (pi_up - pi_down)/(2.0*h);
        return vega;
    }
    
    //helper function for cdf of normal distribution
    static double cdf(double x){
        return 0.5*std::erfc(-x/std::sqrt(2.0)); //(1/2)(errorfunction(-x/sqrt{2})) = Normal_cdf(x)
    }
    //pricing a European call option using the Black-Scholes call option price formula: C = S_0*N(d_1) - (Ke^{-rt})N(d_2), with N = cdf of normal, d1 = (1/\sigma*\sqrt{t})*(ln(\frac{S_0}{K}) + t(r + \frac{\sigma^{2}}{2})) and d2 = (1/\sigma*\sqrt{t})*(ln(\frac{S_0}{K})-t(r+\frac{\sigma^2}{2}))
    double euro_call(const double S0, const double K, const int d, const double sigma, const double r){
        double T = d /255.0;
        double d1 = (1.0/(sigma*std::sqrt(T)))*(std::log(S0/K) + T*(r + 0.5*sigma*sigma));
        double d2 = (1.0/(sigma*std::sqrt(T)))*(std::log(S0/K) + T*(r - 0.5*sigma*sigma));
        return S0*cdf(d1) - K*std::exp(-r*T)*cdf(d2);
    }
    double european_delta(const double S0, const double K, const int d, const double sigma, const double r){
        double T    = d/255.0;
        double sqT  = std::sqrt(T);
        double inv  = 1.0/(sigma*sqT);
        double num  = std::log(S0/K) + (r + 0.5*sigma*sigma)*T;
        double d1   = num * inv;
        return cdf(d1);
    }
private:
    //  overload price() to take a pre‐seeded GSL RNG*
    double price(const long long n, const double S0, const double K, const int d, const double sigma, const double r, gsl_rng *rng)
    {
        double dt = 1.0/255.0;
        double T  = d * dt;
        double totalPayoff = 0.0;

        for(long long i = 0; i < n; ++i) {
            double ST = S0;
            double payoffSum = S0;
            for(int k = 1; k < d; ++k) {
                double z = gsl_ran_gaussian(rng, 1.0);
                ST *= std::exp((r - 0.5*sigma*sigma)*dt + sigma*std::sqrt(dt)*z);
                payoffSum += ST;
            }
            double pathAvg = payoffSum / d;
            totalPayoff += std::max(pathAvg - K, 0.0);
        }

        double sampleAvg = totalPayoff / n;
        return sampleAvg * std::exp(-r*T);
    }
};

int main() {
    // Initialize GSL once
    gsl_rng_env_setup();

    //test parameters
    const long long n      = 1000000;  // # of MC samples
    const double S0        = 100.0;    // spot
    const double K         = 103.0;    // strike
    const int    d         = 255;      // days to expiry
    const double sigma     = 0.10;     // vol
    const double r         = 0.01;     // rate

    /*  Instantiate / calculate */
    Asian asian;
    double asia_npv   = asian.price(n, S0, K, d, sigma, r);
    double asia_delta = asian.delta(n, S0, K, d, sigma, r);
    double vega  = asian.vega(n, S0, K, d, sigma, r);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Asian Option (MC) Results:\n";
    std::cout << "  NPV:   " << asia_npv   << "\n";
    std::cout << "  Delta: " << asia_delta << "\n";
    std::cout << "  Vega:  " << vega  << "\n";
    //compare European option with Asian option:
    double euro_delta = asian.european_delta(S0,K,d,sigma,r);
    double diff = asia_delta - euro_delta;
    std::cout << "European Delta: " << euro_delta << (diff>0 ? " (Asian is +" : " (European is +") << std::abs(diff) << " relative)\n";
    double euro_price = asian.euro_call(S0, K, d, sigma, r);
    std::cout<< "European Option Price: " << euro_price << "\n";
    return 0;
}
