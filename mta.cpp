#include "mta.h"
using namespace std;

MTA::MTA() {
  // No synchronous arcs by default.
  // Call UseFSA for synchronous arcs.
  fsa = false;
}

MTA::~MTA() {
}

void MTA::ComputeComponents() {
  // Find the number of components, as well as the features inside
  // each component.
  for (unsigned int j = 0; j < feature_size.size(); ++j) {
    // A component is uniquely identified by the set of tasks using the component.
    vector<bool> in_task(tasks.size(), 0);
    for (unsigned int i = 0; i < tasks.size(); ++i)
      if (tasks[task_names[i]]->HasFeature(j))
        in_task[i] = 1;

    // Find the component with the same in_task set.
    bool found = false;
    for (unsigned int k = 0; k < components.size(); ++k) {
      if (components[k].in_task == in_task) {
        found = true;
        components[k].features[j] = 1;
        break;
      }
    }

    // If not found, insert this new component.
    if (!found) {
      Component new_component;
      new_component.in_task = in_task;
      new_component.features.resize(feature_size.size(), 0);
      new_component.features[j] = 1;
      components.push_back(new_component);
    }
  }

  // Inform each task the components information and
  // which components are relevant.
  for (unsigned int i = 0; i < tasks.size(); ++i) {
    tasks[task_names[i]]->components.resize(components.size(), 0);
    tasks[task_names[i]]->component_info.resize(components.size());
    for (unsigned int k = 0; k < components.size(); ++k) {
      tasks[task_names[i]]->component_info[k] = &components[k];

      // Check if component k is used in task i,
      // by checking if all the features are there.
      bool relevant = true;
      for (unsigned int j = 0; j < feature_size.size(); ++j) {
        if (components[k].features[j])
          if (!tasks[task_names[i]]->features[j]) {
            relevant = false;
            break;
          }
      }
      if (relevant)
        tasks[task_names[i]]->components[k] = 1;
    }
  }

  // Debug
  /*
  cout << "The components are \n";
  for (auto c : components) {
    cout << "The relevant features are ";
    for (auto f : c.features)
      cout << f << " ";
    cout << "\n";
    cout << "The component is in task ";
    for (auto t : c.in_task)
      cout << t << " ";
    cout << "\n";
  }
  for (unsigned int i = 0; i < tasks.size(); ++ i) {
    cout << "Task " << i << " has components: ";
    for (auto k : tasks[task_names[i]]->components)
      cout << k << " ";
    cout << "\n";
  }
  */
}

void MTA::GenerateContextualDependencyTable() {
  // The contextual dependency table has components.size() rows,
  // and total_actions + 1 columns. The last column is for no-op action.
  cdtb.resize(components.size());
  for (unsigned int k = 0; k < components.size(); ++k) {
    cdtb[k].resize(total_actions + 1);

    // Filling each cell in the table.
    for (int a = 0; a < total_actions; ++a) {
      // Find X_a, the set of tasks with action a.
      vector<bool> X_a(tasks.size(), 0);
      for (unsigned int i = 0; i < tasks.size(); ++i) {
        if (tasks[task_names[i]]->HasAction(a))
          X_a[i] = 1;
      }

      // If FSA is used, parent feature set needs to multiple 2 to include current time step.
      cdtb[k][a].parent_features.resize(fsa ? feature_size.size() * 2 : feature_size.size(), 0);

      // The parent features are features in the task (X_a intersects Y).
      // i.e. the set of tasks that the component is used and has action a.
      // Also calculates the total number of possible parent values.
      int parent_values = 1;
      // Find out the intersection between X_a and Y
      vector<bool> intersection (tasks.size(), 0);
      for (unsigned int i = 0; i < tasks.size(); ++i) {
        intersection[i] = X_a[i] && components[k].in_task[i];
      }

      // If intersection is empty, we use assumption 2 for the no-op action,
      // leaving the cell empty.
      if (accumulate(intersection.begin(), intersection.end(), 0) == 0) {
        continue;
      }

      // Now find all the common features in the task intersection
      for (unsigned int j = 0; j < feature_size.size(); ++j) {
        bool feature_used = true;
        for (unsigned int i = 0; i < tasks.size(); ++i) {
          // If task does not belong to the intersection set of tasks, skip.
          if (!intersection[i])
            continue;
          // Or if the feature does not belong to the task in the intersection, skip.
          if (!tasks[task_names[i]]->features[j])
            feature_used = false;
        }
        if (feature_used) {
          cdtb[k][a].parent_features[j] = true;
          parent_values *= feature_size[j];

          // For FSA, it also depends on this feature at the current step.
          if (fsa) {
            // Include this feature if it belongs to a component with higher order.
            // First find the component this feature belongs to.
            int component_number;
            for (unsigned int k2 = 0; k2 < components.size(); ++k2) {
              if (components[k2].features[j]) {
                component_number = k2;
                break;
              }
            }
            // Check if component_number represents a higher order component than k
            if (IsStrictSubsetOf(components[k].in_task, components[component_number].in_task)) {
              cdtb[k][a].parent_features[j + feature_size.size()] = true;
              parent_values *= feature_size[j];
            }
          }
        }
      }

      // Initializes other members of the cell.
      cdtb[k][a].parent_size = parent_values;
      cdtb[k][a].exploration_count.resize(parent_values, 0);
      cdtb[k][a].distribution.resize(parent_values);
      cdtb[k][a].component = &components[k];

      // Debug
      /*
      cout << "Component " << k << " and action " << a << ": ";
      cout << "has parents value " << cdtb[k][a].parent_size << "\n";
      cout << "The parent features are :";
      for (auto f : cdtb[k][a].parent_features)
        cout << f << " ";
      cout << "\n";
      */

    }

    // The no-op action.
    cdtb[k][total_actions].parent_features = components[k].features;
    if (fsa) {
      cdtb[k][total_actions].parent_features.resize(components[k].features.size() * 2, 0);
    }
    int parent_values = 1;
    for (unsigned int j = 0; j < feature_size.size(); ++j) {
      if (cdtb[k][total_actions].parent_features[j] == 1) {
        parent_values *= feature_size[j];
        // Nothing extra to do for FSA if the action is no-op.
      }
    }
    cdtb[k][total_actions].parent_size = parent_values;
    cdtb[k][total_actions].exploration_count.resize(parent_values, 0);
    cdtb[k][total_actions].distribution.resize(parent_values);
    cdtb[k][total_actions].component = &components[k];

    for (unsigned int i = 0; i < tasks.size(); ++i) {
      tasks[task_names[i]]->cdtb = &cdtb;
      tasks[task_names[i]]->exploration_threshold = exploration_threshold;
    }
  }
}

void MTA::UseFSA() {
  fsa = true;
  for (auto i : tasks)
    i.second->fsa = true;
}
