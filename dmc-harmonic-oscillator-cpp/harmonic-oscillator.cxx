#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>

using namespace std;

std::default_random_engine generator;
std::uniform_real_distribution<double> uniform(0.0,1.0);
std::normal_distribution<double> normal(0.0, 1.0);

typedef float real_t;

real_t ran_uniform() {
    return uniform(generator);
}

real_t ran_gaussian() {
    return normal(generator);
}

const int DIM = 3;

class Walker {
public:
    real_t pos[DIM];
    bool alive = true;
    Walker(real_t posns[DIM]) {
        for(int ii = 0; ii < DIM; ++ii)
            pos[ii] = posns[ii];
    }
};

real_t V(real_t *r) {
    real_t rSqd = 0;
    for (int d = 0; d < DIM; d++)
        rSqd += r[d] * r[d];
    return 0.5 * rSqd;
}

std::vector<Walker> walkers;

void initialize(int N_T) {
    for (int n = 0; n < N_T; n++) {
        real_t p[DIM];
        for(int d = 0; d < DIM; d++)
            p[d] = ran_uniform() - 0.5;
        walkers.push_back(Walker(p));
    }
}

void oneMonteCarloStep(int n, real_t dt, real_t E_T) {
    for(int d = 0; d < DIM; d++)
        walkers[n].pos[d] += ran_gaussian() * sqrt(dt);

    real_t dv = V(walkers[n].pos) - E_T;
    real_t q = exp(- dt * dv);
    real_t u = ran_uniform();

    int survivors = int(q);
    if(q - survivors > u)
        ++survivors;

    int N = walkers.size();
    for(int i = 0; i < survivors - 1; i++) {
        walkers.push_back(Walker(walkers[n].pos));
    }

    if(survivors == 0)
        walkers[n].alive = false;
}

real_t ESum = 0, ESqdSum = 0;
real_t E_T = 0;

void oneTimeStep(real_t dt, int N_T) {
    int N_0 = walkers.size();

    for(int n = 0; n < N_0; n++)
        oneMonteCarloStep(n, dt, E_T);

    int newN = 0;
    for (int n = 0; n < walkers.size(); n++) {
        if(walkers[n].alive) {
            if(n != newN) {
                for (int d = 0; d < DIM; d++)
                    walkers[newN].pos[d] = walkers[n].pos[d];
                walkers[newN].alive = true;
            }
            ++newN;
        }
    }
    N = newN;

    E_T += log(N_T / real_t(N)) / 10;
//    cout << E_T << "\n";

    ESum += E_T;
    ESqdSum += E_T * E_T;
}

int main() {
    int N_T;
    real_t dt;
    cout << " Diffusion Monte Carlo for the 3-D Harmonic Oscillator\n"
         << " -----------------------------------------------------\n";
    cout << " Enter desired target number of walkers: ";
    cin >> N_T;
    cout << " Enter time step dt: ";
    cin >> dt;

    cout << " Enter total number of time steps: ";
    int timeSteps;
    cin >> timeSteps;
    initialize(N_T);

    for (int i = 0; i < timeSteps; i++) {
        oneTimeStep(dt, N_T);

    }
// compute averages
    real_t EAve = ESum / timeSteps;
    real_t EVar = ESqdSum / timeSteps - EAve * EAve;
    cout << " <E> = " << EAve << " +/- " << sqrt(EVar / timeSteps) << endl;
    cout << " <E^2> - <E>^2 = " << EVar << endl;
    return 0;
}
