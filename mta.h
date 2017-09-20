#include <vector>
#include <map>
#include <iostream>
#include <string>
#include "task.h"
#include "Utility.h"

using namespace std;

// The class that contains multiple tasks, generates the contextual dependency
// table, initialize all components, etc.
class MTA {
 public:
  MTA();
  virtual ~MTA();
  virtual void InitializeTasks() = 0;
  // Compute the value of each components after the tasks are initialized.
  void ComputeComponents();
  void GenerateContextualDependencyTable();
  virtual void GenerateRewardFunction(Task* some_task) = 0;
  virtual void UpdateWithNewObservation(const vector<int>& last_state,
      int action, const vector<int>& curr_state, int reward) = 0;

  // Use FSA. Call this function when the problem has synchronous arcs.
  // This function double the feature_size vector to include current step.
  // Only call the function after feature_size is initialized.
  void UseFSA();

  vector<string> task_names;
  map<string, Task*> tasks;
  // Contextual Dependency Table
  vector<vector<Distribution> > cdtb;
  vector<Component> components;

  // The size of each feature
  vector<int> feature_size;

  int total_actions;

  int exploration_threshold;

  // Uses full synchronous arcs.
  // Set to false by default.
  bool fsa;
};
