#include <iostream>
#include <vector>
#include <stdio.h>
#include <map>
#include <random>
#include <algorithm>
#include <math.h>
#include <float.h>

#define MAX_CARS 20
#define MAX_MOVE_OF_CARS 5
#define RENTAL_REQUEST_FIRST_LOC 3
#define RENTAL_REQUEST_SECOND_LOC 4
#define RETURNS_FIRST_LOC 3
#define RETURNS_SECOND_LOC 2
#define DISCOUNT 0.9
#define RENTAL_CREDIT 10
#define MOVE_CAR_COST 2
#define POISSON_UPPER_BOUND 11

std::map<int,double> poisson_cach;

double poisson_probability(int n, int lam) {
    int key = n * 10 + lam;
    if (::poisson_cach.find(key) == ::poisson_cach.end()) {
        double value = (exp(-lam) * pow(lam,n))/tgamma(n+1);
        ::poisson_cach.insert({key,value});
    }
    return ::poisson_cach[key];
}

double expected_return(int fir, int sec, int action, const std::vector<std::vector<double>> &state_value) {
    double returns = 0.0;

    returns -= MOVE_CAR_COST * std::abs(action);

    int NUM_OF_CARS_FIRST_LOCATION = std::min(fir - action, MAX_CARS);
    int NUM_OF_CARS_SECOND_LOC = std::min(sec + action, MAX_CARS);

    for (int i = 0; i < POISSON_UPPER_BOUND; i++) {
        for (int j = 0; j < POISSON_UPPER_BOUND; j++) {
            double prob = poisson_probability(i, RENTAL_REQUEST_FIRST_LOC) * poisson_probability(j,RENTAL_REQUEST_SECOND_LOC);


            int num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOCATION;
            int num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC;

            int valid_rental_first_loc = std::min(num_of_cars_first_loc, i);
            int valid_rental_second_loc = std::min(num_of_cars_second_loc, j);

            double reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT;
            num_of_cars_first_loc -= valid_rental_first_loc;
            num_of_cars_second_loc -= valid_rental_second_loc;

            int returned_cars_first_loc = RETURNS_FIRST_LOC;
            int returned_cars_second_loc = RETURNS_SECOND_LOC;

            num_of_cars_first_loc = std::min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS);
            num_of_cars_second_loc = std::min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS);

            returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc][num_of_cars_second_loc]);
        }
    }
    
    return returns;
}

double max_value_change(const std::vector<std::vector<double>> &old_value, const std::vector<std::vector<double>> &new_value) {
    double max = -__DBL_MAX__;
    for (int i = 0; i < old_value.size(); i++) {
        for (int j = 0; j < old_value[0].size(); j++) {
            double sub = new_value[i][j] - old_value[i][j];
            //std::cout<<old_value[i][j]<<std::endl;
            if (max < sub) {
                max = sub;
            }
        }
    }
    //std::cout<<max<<std::endl;
    return std::abs(max);
}


int main() {
    std::vector<std::vector<double>> value(MAX_CARS+1, std::vector<double>(MAX_CARS+1,0));
    std::vector<std::vector<double>> policy(MAX_CARS+1, std::vector<double>(MAX_CARS+1,0));
    std::vector<int> actions = {-5,-4,-3,-2,-1,0,1,2,3,4,5};

    int policy_iterations = 1;

    while (true) {
        while (true) {
            auto old_value(value);
            for (int i = 0; i < MAX_CARS+1; i++) {
                for (int j = 0; j < MAX_CARS+1; j++) {
                    value[i][j] = expected_return(i,j,policy[i][j],value);
                    //std::cout<<value[i][j]<<std::endl;
                }
            }
            double max_change = max_value_change(old_value,value);
            printf("Max value change %.17g\n", max_change);
            if (max_change < 1e-4) {
                break;
            }
        }

        std::cout<<"Policy iteration: "<<policy_iterations++<<std::endl;
        bool policy_stable = true;
        for (int i = 0; i < MAX_CARS+1; i++) {
            for (int j = 0; j < MAX_CARS+1; j++) {
                int old_action = policy[i][j];
                /*
                std::vector<double> action_returns;
                for (int index_action = 0; index_action < actions.size(); index_action++) {
                    if (actions[index_action] >= -j && actions[index_action] <= i) {
                        action_returns.push_back(expected_return(i,j,actions[index_action],value));
                    } else {
                        action_returns.push_back(-__DBL_MAX__);
                    }
                }
                
                policy[i][j] = actions[std::distance(action_returns.begin(),std::max_element(action_returns.begin(),action_returns.end()))];
                */
               
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
                policy[i][j] = actions[index];
                
                if (policy_stable && old_action != policy[i][j]) {
                    policy_stable = false;
                }
            }
        }

        if (policy_stable) {
            break;
        }
    }

    for (int i = 0; i < policy.size(); i++) {
        for (int j = 0; j < policy[0].size(); j++) {
            std::cout<<policy[i][j]<<" ";
        }
        std::cout<<std::endl;
    }

    return 0;
}
