#ifndef __VALUEITERATION_H
#define __VALUEITERATION_H

#include <vector>
#include <string>

/**
   @class ValueIteration
   @brief Simple value iteration for MDPs
   @details 
   @author Wee Sun Lee
   @date 26 October 2009

   @modified by Li Zhuoru
   @date 2014-2015
*/

using namespace std;

class ValueIteration
{
 public:
  ValueIteration(long numStates, long numActions, double discount, vector<double>& values):
      values(values), numStates(numStates), numActions(numActions), discount(discount) {
    actionApplicable.resize(numStates);
    for (int i = 0; i < numStates; ++i)
      actionApplicable[i].resize(numActions, true);
  };

  ValueIteration(long numStates, long numActions, double discount, const vector<vector<bool> >& actionApplicable, vector<double>& values):
     values(values), numStates(numStates), numActions(numActions), discount(discount), actionApplicable(actionApplicable) {};


    void doValueIteration(std::vector<std::vector<double> >& rewardMatrix, std::vector<std::vector<std::vector<std::pair<long,double> > > >& transMatrix, double targetPrecision, long displayInterval = 100);
    
    std::vector<double> values;
    std::vector<int> actions;
    
    /** 
      Write out the policy \a filename
    */
    void write(std::string filename);
    
    /**
       Reads in what has be written out with \a write
    */
    void read(std::string filename);

 private:
    long numStates;
    long numActions;
    double discount;
    vector<vector<bool> > actionApplicable;

};

#endif // __VALUEITERATION_H
