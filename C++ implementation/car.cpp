#include <iostream>
#include <vector>
#include <stdio.h>
#include <map>
#include <random>
#include <algorithm>
#include <math.h>

// max number of cars for location
#define MAX_CARS 20
// max number of cars that it is possible to move in one night
#define MAX_MOVE_OF_CARS 5
// mean of requested cars in the first location
#define RENTAL_REQUEST_FIRST_LOC 3
// mean of requested cars in the second location
#define RENTAL_REQUEST_SECOND_LOC 4
// mean of returned cars in the first location
#define RETURNS_FIRST_LOC 3
// mean of returned cars in the second location
#define RETURNS_SECOND_LOC 2
// discount factor
#define DISCOUNT 0.9
// cost of reting a car
#define RENTAL_CREDIT 10
// cost of moving a car from one location to another
#define MOVE_CAR_COST 2
// max number of rented cars in a single day
#define POISSON_UPPER_BOUND 11

// global variable to store probability
std::map<int,double> poisson_cach;

// funtion that compute the probability with the pmf poisson
double poisson_probability(int n, int lam) {
    int key = n * 10 + lam;
    if (::poisson_cach.find(key) == ::poisson_cach.end()) {
        double value = (exp(-lam) * pow(lam,n))/tgamma(n+1);
        ::poisson_cach.insert({key,value});
    }
    return ::poisson_cach[key];
}

double expected_return(int fir, int sec, int action, const std::vector<std::vector<double>> &state_value) {
    // initialize the variable to return
    double returns = 0.0;

    // substract cost of moving cars
    returns -= MOVE_CAR_COST * std::abs(action);

    // update the number of cars in each location
    int NUM_OF_CARS_FIRST_LOCATION = std::min(fir - action, MAX_CARS);
    int NUM_OF_CARS_SECOND_LOC = std::min(sec + action, MAX_CARS);

    // for every possible case of rental request
    for (int i = 0; i < POISSON_UPPER_BOUND; i++) {
        for (int j = 0; j < POISSON_UPPER_BOUND; j++) {
            // compute the joint probability
            double prob = poisson_probability(i, RENTAL_REQUEST_FIRST_LOC) * poisson_probability(j,RENTAL_REQUEST_SECOND_LOC);

            int num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOCATION;
            int num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC;

            // get valid number of rental request
            int valid_rental_first_loc = std::min(num_of_cars_first_loc, i);
            int valid_rental_second_loc = std::min(num_of_cars_second_loc, j);

            // compute credit for renting cars
            double reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT;

            // update number of cars
            num_of_cars_first_loc -= valid_rental_first_loc;
            num_of_cars_second_loc -= valid_rental_second_loc;

            // get number of returned cars
            int returned_cars_first_loc = RETURNS_FIRST_LOC;
            int returned_cars_second_loc = RETURNS_SECOND_LOC;

            // get valid number of final cars in each location
            num_of_cars_first_loc = std::min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS);
            num_of_cars_second_loc = std::min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS);

            // bellman equation
            returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc][num_of_cars_second_loc]);
        }
    }
    
    return returns;
}

// return max variation in the two matrices
double max_value_change(const std::vector<std::vector<double>> &old_value, const std::vector<std::vector<double>> &new_value) {
    double max = -__DBL_MAX__;
    for (int i = 0; i < old_value.size(); i++) {
        for (int j = 0; j < old_value[0].size(); j++) {
            double sub = new_value[i][j] - old_value[i][j];
            if (max < sub) {
                max = sub;
            }
        }
    }
    return std::abs(max);
}


int main() {
    // value estimation for every state
    std::vector<std::vector<double>> value(MAX_CARS+1, std::vector<double>(MAX_CARS+1,0));

    // policy for every state
    std::vector<std::vector<double>> policy(MAX_CARS+1, std::vector<double>(MAX_CARS+1,0));

    // possible actions
    std::vector<int> actions = {-5,-4,-3,-2,-1,0,1,2,3,4,5};

    int policy_iterations = 1;
    
    while (true) {
        while (true) {
            // save matrix of value of previous iteration
            auto old_value(value);

            for (int i = 0; i < MAX_CARS+1; i++) {
                for (int j = 0; j < MAX_CARS+1; j++) {
                    value[i][j] = expected_return(i,j,policy[i][j],value);
                }
            }

            // get max change of value estimation and print it
            double max_change = max_value_change(old_value,value);
            printf("Max value change %.17g\n", max_change);

            if (max_change < 1e-4) {
                break;
            }
        }

        // print the number of time the policy is updated
        std::cout<<"Policy iteration: "<<policy_iterations++<<std::endl;

        bool policy_stable = true;
        for (int i = 0; i < MAX_CARS+1; i++) {
            for (int j = 0; j < MAX_CARS+1; j++) {

                // save old policy for the state
                int old_action = policy[i][j];

                // get index of action with maximum expected return
                double max = -__DBL_MAX__;
                double index = -1;
                for (int index_action = 0; index_action < actions.size(); index_action++) {
                        int action = actions[index_action];
                        if ((action >= 0 && action <= i) || (action >= -j && action <= 0)) {
                            double tmp = expected_return(i,j,action,value);
                            if (max < tmp) {
                                max = tmp;
                                index = index_action;
                            }
                        }
                }

                // update policy
                policy[i][j] = actions[index];
                
                // check if the policy changed
                if (policy_stable && old_action != policy[i][j]) {
                    policy_stable = false;
                }
            }
        }

        if (policy_stable) {
            break;
        }
    }

    // print the policy matrix
    for (int i = 0; i < policy.size(); i++) {
        for (int j = 0; j < policy[0].size(); j++) {
            std::cout<<policy[i][j]<<" ";
        }
        std::cout<<std::endl;
    }

    return 0;
}
