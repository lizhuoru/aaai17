#include "task.h"
#include "Utility.h"
#include <cmath>

// This function updates each entry in the contextual dependency table with
// new observation.
void Distribution::UpdateWithNewExperience(const vector<int>& last_state,
    const vector<int>& current, const vector<int>& feature_size, bool fsa) {
  vector<int> parent_feature_size = feature_size;

  // Find the integer representation of the parent feature.
  int parent;
  if (!fsa)
    parent = MapFactoredStateToInt(last_state, feature_size, parent_features);
  else
    parent = CheckAndMapParentFSA(last_state, current, feature_size, parent_features);

  // Find the integer representation of the component value.
  int child = MapFactoredStateToInt(current, feature_size, component->features);

  // Update Probability
  bool found = false;
  for(vector<pair<long, double> >::iterator it = distribution[parent].begin();
      it != distribution[parent].end(); ++it) {
    if (it->first != child) {
      it->second = it->second * (double(exploration_count[parent]) /
         (exploration_count[parent] + 1));
    }
    else if (it->first == child) {
      it->second = (it->second * double(exploration_count[parent]) + 1) /
        (exploration_count[parent] + 1);
      found = true;
    }
  }
  // Increment the visit count
  exploration_count[parent]++;
  if (!found) {
    // Add a new entry
    distribution[parent].push_back(make_pair(child, 1.0/exploration_count[parent]));
  }
}

Task::Task(const vector<bool>& features, const vector<bool>& actions, string name,
    const vector<int>& feature_size, int rmax):
    features(features),
    actions(actions),
    task_name(name),
    feature_size(feature_size),
    rmax(rmax) {

  total_actions = accumulate(actions.begin(), actions.end(), 0);
  total_steps = 0;
  state_size = 1;
  for (unsigned int j = 0; j < features.size(); ++j) {
    if (features[j]) {
      state_size *= feature_size[j];
    }
  }


  // Includes fictitious state.
  transition.resize(state_size + 1);
  reward.resize(state_size + 1);
  applicable_actions.resize(state_size + 1);
  values.resize(state_size + 1, rmax/0.1);

  for (int s = 0; s < state_size; ++s) {
    transition[s].resize(total_actions);
    // Reward initialize to rmax.
    reward[s].resize(total_actions, rmax);
    // By default every action is available.
    applicable_actions[s].resize(total_actions, true);

    // Initial state value for value iteration.
    values[s] = rmax/0.1;
  }

  // Initialize for the fictitious state.
  transition[state_size].resize(total_actions);
  reward[state_size].resize(total_actions, rmax);
  applicable_actions[state_size].resize(total_actions, true);

  // Contextual dependency table is initialized later by MTA class.
  cdtb = 0;
  vi = new ValueIteration(state_size + 1, total_actions, 0.9, applicable_actions, values);

  fsa = false;
}

Task::~Task() {
  delete vi;
}

int Task::MapGlobalToLocal(const int global, const vector<bool>& bit_map) {
  if (!bit_map[global]) {
    cerr << "Not applicable to this task!\n";
    return -1;
  }
  if (bit_map.size() > static_cast<unsigned int>(global)) {
    return accumulate(bit_map.begin(),
        bit_map.begin() + global, 0);
  }
  cerr << "Cannot map global to local! Global is " << global << "\n";
  cerr << "Bit_map is ";
  for (auto i : bit_map) {
    cerr << i;
  }
  cerr << "\n";
  return -1;
}

int Task::MapLocalToGlobal(const int local,
    const vector<bool>& bit_map) {
  int counter = -1;
  for (unsigned int i = 0; i < bit_map.size(); ++i) {
    counter += bit_map[i];
    if (counter == local)
      return i;
  }
  cerr << "Cannot map local to global!\n";
  return -1;
}

// This function should only be called after the contextual
// dependency table is constructed.
void Task::ConstructTransitionFunction() {
  if (fsa) {
    ConstructTransitionFunctionFSA();
    return;
  }

  // Iterate over all states.
  for(int i = 0; i < state_size; ++i) {
    // Looping through the contextual dependency table.
    // Each action is a different column in the table.
    for (int a = 0; a < total_actions; ++a) {
      FindNextStates(i, a);
    }
  }

  // Constructing the transition function for the fictitious state.
  for (int a = 0; a < total_actions; ++a) {
    // Transit to itself.
    transition[state_size][a].resize(0);
    transition[state_size][a].push_back(make_pair(state_size, 1.0));
    reward[state_size][a] = rmax;
  }
}

void Task::FindNextStates(int state, int action) {
  vector<int> current_state;
  MapIntStateToVector(state, feature_size, features, current_state);

  // Total number of components used by this task.
  total_components = accumulate(components.begin(), components.end(), 0);
  // Iterate over |a| columns of k rows in the contextual dependency table.
  // This records the position of iteration.
  vector<int> counter(total_components, 0);
  // action must be converted to global index to access contextual dependency table.
  int global_a = MapLocalToGlobal(action, actions);

  vector<int> component_order;
  ComputeOrderFSA(component_order);

  // Reset all contents in the transition function.
  transition[state][action].resize(0);

  // Different components have different parents, thus they need to be computed.
  vector<int> parents(total_components, 0);

  // Constructing transition function for state i action a.
  bool terminate = false;
  double probability;

  // Reset the counter.
  fill(counter.begin(), counter.end(), 0);

  // If exploration count is less than threshold, set the flag to true.
  bool fictitious_state_flag = false;
  while (!terminate) {
    vector<int> next_state(features.size(), -1);
    // Fill in next_state.
    probability = 1.0;
    for (int l = 0; l < total_components; ++l) {
      int k = component_order[l];
      // k is a local component index and thus needs to be converted to global index
      // to access contextual dependency table.
      int global_k = MapLocalToGlobal(k, components);
      if (!fsa)
        parents[k] = MapFactoredStateToInt(current_state, feature_size,
                  (*cdtb)[global_k][global_a].parent_features);
      else
        parents[k] = CheckAndMapParentFSA(current_state, next_state, feature_size,
                  (*cdtb)[global_k][global_a].parent_features);

      // If the exploration threshold is not reached, transit to the fictitious state;
      if ((*cdtb)[global_k][global_a].exploration_count[parents[k]] < exploration_threshold) {
        fictitious_state_flag = true;
        break;
      }

      // Debug
      /*
      cout << "parents[k] is " << parents[k] << "\n";
      cout << (*cdtb)[global_k][global_a].distribution.size() << "\n";
      cout << "counter[l] is " << counter[l] << "\n";
      cout << (*cdtb)[global_k][global_a].distribution[parents[k]].size() << "\n";
      */

      int component_value = (*cdtb)[global_k][global_a].distribution[parents[k]][counter[l]].first;
      vector<int> component;
      MapIntStateToVector(component_value, feature_size,
          component_info[global_k]->features, component);

      // Debug
      /*
      cout << "This component has value " << component_value << "\n";
      cout << "This component has features ";
      for (auto j : component_info[global_k]->features)
        cout << j << " ";
      cout << "\n";
      cout << "This component has values ";
      for (auto j : component)
        cout << j << " ";
      cout << "\n";
      */

      // Combining component features to form the next state
      for (unsigned int m = 0; m < features.size(); ++m) {
        if (component_info[global_k]->features[m]) {
          next_state[m] = component[m];
        }
      }

      probability *= (*cdtb)[global_k][global_a].distribution[parents[k]][counter[l]].second;
    }

    // Skip constructing transition function
    if (fictitious_state_flag) {
      break;
    }

    // Debug
    /*
    cout << "Next state is ";
    for (auto j : next_state)
      cout << j << " ";
    cout << "with probability " << probability << "\n";
    */

    transition[state][action].push_back(make_pair(MapFactoredStateToInt(next_state,
            feature_size, features), probability));

    // Increment Counter. Starting from the last counter.
    counter[total_components-1]++;
    for (int l = total_components -1; l > 0; l--) {
      int k = component_order[l];
      int global_k = MapLocalToGlobal(k, components);
      counter[l-1] += counter[l]/(*cdtb)[global_k][global_a].distribution[parents[k]].size();
      counter[l] %= (*cdtb)[global_k][global_a].distribution[parents[k]].size();
    }

    // Check if first counter has reached the end.
    int first = MapLocalToGlobal(component_order[0], components);
    if (static_cast<unsigned int>(counter[0])
        >= (*cdtb)[first][global_a].distribution[parents[component_order[0]]].size()) {
      terminate = true;
      break;
    }
  }

  if (fictitious_state_flag) {
    // Transit to fictitious state with probability 1.
    // The fictitious state has an index of "state_size".
    transition[state][action].resize(0);
    transition[state][action].push_back(make_pair(state_size, 1.0));
    reward[state][action] = rmax;
  }
}

void Task::ComputeOrderFSA(vector<int>& component_order) {
  // Order the components by the number of tasks.
  // Components used by more tasks will be evaluated first.
  // Loop down from the number of tasks to 0.
  for (int i = component_info[0]->in_task.size(); i > 0; --i) {
    for (int k = 0; k < total_components; ++k) {
      int global_k = MapLocalToGlobal(k, components);
      if (accumulate(component_info[global_k]->in_task.begin(),
            component_info[global_k]->in_task.end(), 0) == i) {
        component_order.push_back(k);
      }
    }
  }

  // Debug
  /*
  cout << "The component order is \n";
  for (auto k : component_order)
    cout << k << " ";
  cout << "\n";
  */

}

// The FSA version of transition function construction.
void Task::ConstructTransitionFunctionFSA() {
  // Total number of components used by this task.
  total_components = accumulate(components.begin(), components.end(), 0);

  // Iterate over |a| columns of k rows in the contextual dependency table.
  // This records the position of iteration.
  vector<int> counter(total_components, 0);

  // Iterate over all states.
  for(int i = 0; i < state_size; ++i) {
    vector<int> current_state;
    MapIntStateToVector(i, feature_size, features, current_state);

    // Looping through the contextual dependency table.
    // Each action is a different column in the table.
    for (int a = 0; a < total_actions; ++a) {
      FindNextStates(i, a);
    }
  }

  // Constructing the transition function for the fictitious state.
  for (int a = 0; a < total_actions; ++a) {
    // Transit to itself.
    transition[state_size][a].resize(0);
    transition[state_size][a].push_back(make_pair(state_size, 1.0));
    reward[state_size][a] = rmax;
  }
}

int Task::SelectBestAction(const vector<int>& current_state, bool speedup) {
  if (speedup == true) {
    // If any component action pair is not sufficiently explored, just execute this action
    for (int k = 0; k < total_components; ++k) {
      int global_k = MapLocalToGlobal(k, components);
      int parent_k = MapFactoredStateToInt(current_state, feature_size, component_info[global_k]->features);
      for (int a = 0; a < total_actions; ++a) {
        int global_a = MapLocalToGlobal(a, actions);
        if ((*cdtb)[global_k][global_a].exploration_count[parent_k] < exploration_threshold) {
          return global_a;
        }
      }
    }

    // Else run vi for every 50 steps. For the other steps, just use the old policy.
    int curr = MapFactoredStateToInt(current_state, feature_size, features);
    if (total_steps % 50 != 0) {
      int a = vi->actions[curr];
      return MapLocalToGlobal(a, actions);
    }
  }

  vi -> doValueIteration(reward, transition, 0.1);
  int s = MapFactoredStateToInt(current_state, feature_size, features);
  int best_action = vi->actions[s];

  // The action returned should be converted to global index.
  int global_action = MapLocalToGlobal(best_action, actions);

  // Debug
  /*
  for (int s = 0; s <= state_size; ++s) {
    cout << "Printing transition function for state " << s << "\n";
    for (int a =0; a < total_actions; ++a) {
      cout << "Printing transition function for action " << a << "\n";
      for (auto j : transition[s][a]) {
        cout << j.first << " " << j.second << " ";
      }
      cout << "\n";
    }
  }

  cout << "\n";
  cout << "Printing the transition function for last state\n";
  for (int a = 0; a < total_actions; ++a) {
    cout << "action is " << a << "\n";
    for (auto i : transition[s][a]) {
      cout << i.first << " " << i.second << "\n";
    }
  }

  for (int s = 0; s <= state_size; ++s) {
    cout << "Value for state " << s << " is " << values[s] << "\n";
    cout << "Applicable actions are ";
    for (auto a : applicable_actions[s])
      cout << a << " ";
    cout << "\n";
  }
  cout << "\n";

  cout << "Current state is " << s << ". ";
  cout << "Action selected is " << global_action << "\n";
  cout << "Current value is " << values[s] << "\n";

  cout << "Reward is ";
  for (auto i : reward[s])
    cout << i << " ";
  cout << "\n";

  */

  total_steps++;
  return global_action;
}
