#include "ValueIteration.h"
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

void ValueIteration::doValueIteration(std::vector<std::vector<double> >& rewardMatrix, std::vector<std::vector<std::vector<std::pair<long,double> > > >& transMatrix, double targetPrecision, long displayInterval)
{
  // record time
  time_t start, curr;
  double timeSoFar = 0;

  // Allocate memory for values and actions
  // Question: What are values and actions?
  values.resize(numStates);
  actions.resize(numStates);

  time(&start);
  time(&curr);

  vector<vector<double> > tempValues;

  // What are tempValues?
  // It's the temp values to compare changes
  // Note: Two of them initialized to all 0.
  tempValues.resize(2);
  for (long i=0; i< 2; i++){
    tempValues[i] = values;
  }

  double currChange = FLT_MAX; 
  long currIndex = 0, nextIndex = 1;

  while (currChange > targetPrecision){
    // for display
    double temp = difftime(curr,start);
    if (temp - timeSoFar >= displayInterval){
      timeSoFar = temp;
      //cout << "time: " << temp << " Diff: " << currChange << "\n";
    }
 
    // For each state, look for the best value that goes with best action in that state
    for (long i=0; i< numStates; i++){
      double bestValue = -FLT_MAX;
      long bestAction = 0;
      
      for (long j = 0; j < numActions; j++){

        // Action j is not available for this state i
        if (!actionApplicable[i][j])
          continue;

        // Compute discounted reward
        double currValue = rewardMatrix[i][j];
        for (long k = 0; k < transMatrix[i][j].size(); k++){
          long nextState = transMatrix[i][j][k].first;
          double prob = transMatrix[i][j][k].second;
          currValue +=  discount * prob * tempValues[nextIndex][nextState];
        }
        
        // Seach for best discounted rewards among all actions in this state
        if (currValue > bestValue){
          bestValue = currValue;
          bestAction = j;
        }
      }
      
      // For this iteration tempValues store the best value of the state thus far
      tempValues[currIndex][i] = bestValue;
      actions[i] = bestAction;
    }
    currChange = 0;
    for (long i = 0; i < numStates; i++){
      if (fabs(tempValues[currIndex][i]- tempValues[nextIndex][i])> currChange)
        currChange = fabs(tempValues[currIndex][i]- tempValues[nextIndex][i]);
    }

    currIndex = nextIndex;
    nextIndex = (nextIndex + 1) % 2;
    //cout << " Diff: " << currChange << "\n";
  }

  for (long i =0; i< numStates; i++){
    values[i] = tempValues[nextIndex][i];
  }

  // currChange should grows to 0.
  //cout << "time: " << difftime(curr,start) << " Diff: " << currChange << "\n";
};

void ValueIteration::write(std::string filename)
{
  ofstream fp;
  fp.open(filename.c_str(), ofstream::trunc);
  if (!fp.is_open()){
    cerr << "Fail to open " << filename << "\n";
    exit(EXIT_FAILURE);
  }
  fp << numStates << "\n";
  for (long i=0; i < numStates; i++){
    fp << actions[i] << "\n";
  }
  fp.close();
};

void ValueIteration::read(std::string filename)
{
  ifstream fp; 
  fp.open(filename.c_str(), ios::in);
  if (!fp.is_open()){
    cerr << "Fail to open " << filename << "\n";
    exit(EXIT_FAILURE);
  }

  fp >> numStates;
  actions.resize(numStates,0);
  for (long i=0; i< numStates; i++){
    fp >> actions[i];
  }
  fp.close();
};

