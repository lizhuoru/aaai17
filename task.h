#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include "ValueIteration.h"

using namespace std;

class Component {
 public:
  // If the component is in task j, then the jth term is 1.
  // Note that a component can be uniquely identified by this.
  vector<bool> in_task;
  // List of features contained in this component.
  vector<bool> features;
};

// A distribution maps some parent values to component values.
// Conditional distribution of component values given the parents.
class Distribution {
 public:
  // Stores the distribution
  // First vector is the parent, second vector is the actual distribution
  // given the parent.
  vector<vector<pair<long, double> > > distribution;

  // Stores the exploration count.
  // Its size should be set by the MTA-FRMAX algorithm.
  vector<int> exploration_count;

  // This is also the size of exploration_count vector.
  // Number of values the parent features can take.
  int parent_size;

  // The parent features of the distribution. 1 represents being used.
  vector<bool> parent_features;
  // Provides method to update with new experience.
  void UpdateWithNewExperience(const vector<int>& last_state,
      const vector<int>& current, const vector<int>& feature_size, bool fsa = false);
  // The values of the component it represents.
  Component* component;
};

class Task {
 public:
  Task(const vector<bool>& features, const vector<bool>& actions, string name,
       const vector<int>& feature_size, int rmax);
  ~Task();

  // Could be a task or a task element
  bool IsTask();
  bool HasFeature(int index) {return features[index];};

  // Value 1 represents feature/action is relevant
  // Value 0 represents feature/action is irrelevant
  vector<bool> features;
  vector<bool> actions;
  int total_actions;
  bool HasAction(int a) {return actions[a];};

  string task_name;

  // The number of values each feature can take
  vector<int> feature_size;
  // Total number of states
  int state_size;

  // Set of all components used. Only filled after all tasks are known.
  // 1 represent the component being used.
  vector<bool> components;
  int total_components;
  // The info of each component.
  vector<Component*> component_info;

  bool is_task;

  // Task Transition Function
  vector<vector<vector<pair<long, double> > > > transition;
  // Task Reward Function
  vector<vector<double> > reward;
  // The maximum reward assigned by rmax
  int rmax;

  // Total number of steps has been executed in this task.
  int total_steps;

  // Action 0 for a task is not necessarily action 0 for the problem.
  // Map component/action from global index to local index.
  int MapGlobalToLocal(const int global, const vector<bool>& global_list);
  // Map component/action from local index to global index.
  int MapLocalToGlobal(const int local, const vector<bool>& global_list);

  // This constructs task transition function including fictitious state.
  // It does not construct the full reward function except setting the
  // fictitious states related transition to Rmax reward.
  const vector<vector<Distribution> >* cdtb;
  int exploration_threshold;

  void ConstructTransitionFunction();
  void ConstructTransitionFunctionFSA();
  void FindNextStates(int state, int action);

  // FSA may not execute in order. Check thesis for this section.
  void ComputeOrderFSA(vector<int>& component_order);

  // Solve the task MDP using value iteration
  // If speedup is true, then reduces the frequency of running VI.
  int SelectBestAction(const vector<int>& current_state, bool speedup = false);

  // Not all actions are available at every state.
  // Set to false for non-applicable actions.
  vector<vector<bool> > applicable_actions;

  // This is the value of the task states.
  vector<double> values;

  // The pointer is freed in the destructor.
  ValueIteration* vi;
  bool fsa;
};

