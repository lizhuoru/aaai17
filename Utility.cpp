
#include "Utility.h"
#include <cassert>

int randInRange(int max){
  double r, x;
  r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
  x = (r * (max+1));
  return (int)x;
}

int CheckAndMapParentFSA(const vector<int>& current_state,
    const vector<int>& next_state, const vector<int>& feature_size,
    const vector<bool>& parent_features) {

  vector<int> concat_state = current_state;
  std::copy(next_state.begin(), next_state.end(), std::back_inserter(concat_state));

  // Concatenate parent_feature_size to include current time step.
  vector<int> parent_feature_size = feature_size;
  std::copy(feature_size.begin(), feature_size.end(), std::back_inserter(parent_feature_size));

  // Debug
  /*
  cout << "Current state is: ";
  for (auto j : current_state)
    cout << j << " ";
  cout << "\n";

  cout << "Next state is: ";
  for (auto j : next_state)
    cout << j << " ";
  cout << "\n";

  cout << "concat state is: ";
  for (auto j : concat_state)
    cout << j << " ";
  cout << "\n";

  cout << "parent feature size is: ";
  for (auto j : parent_feature_size)
    cout << j << " ";
  cout << "\n";

  cout << "parent feature is: ";
  for (auto j : parent_features)
    cout << j << " ";
  cout << "\n";
  */

  // Checks if every parent_feature is defined.
  for (unsigned int j = 0; j < parent_features.size(); ++j) {
    if (concat_state[j] == -1) {
      if (parent_features[j]) {
        cerr << "Something is wrong in FSA parent checking\n";
        cerr << "-1 is not a valid value for a feature \n";
        for (auto j : parent_features)
          cout << j << " ";
        cout << "\n";
        assert(!parent_features[j]);
        return -1;
      } else {
        // This is a bit confusing.
        // The MapFactoredStateToInt function does not work with value -1.
        // Hence change it to 0.
        concat_state[j] = 0;
      }
    }
  }

  // Compute parent value
  return MapFactoredStateToInt(concat_state, parent_feature_size, parent_features);
}

// Maps a relevant subset of state variables to flat states
int MapFactoredStateToInt(const vector<int>& state,
    const vector<int>& size, const vector<bool>& relevant) {
  assert(state.size() == size.size());
  assert(state.size() == relevant.size());
  int multiplier = 1;
  int pos = 0;
  int result = 0;

  for(int i = state.size() - 1; i >= 0; --i) {
    // Assuming all state variable starts from 0
    // -1 stands for state variable have no value
    if ((relevant[i]) && (state[i] == -1)) {
      std::cerr << "Error in MapFactoredStateToInt!\n";
      return -1;
    }

    if ((relevant[i]) && (state[i] != -1)) {
      pos = state[i];
      result += pos*multiplier;
      multiplier*=size[i];
    }
  }
  return result;
}

//Maps a flat state to a state vector, filling the relevant features
void MapIntStateToVector(int flat_state,
    const vector<int>& size, const vector<bool>& relevant, vector<int>& result) {
  assert(size.size() == relevant.size());
  int multiplier = 1;
  int next_multiplier;
  int temp;

  // -1 stands for state variable have no value
  // state variable value starts from 0
  result.resize(relevant.size(), -1);

  for(int i = relevant.size() - 1; i >= 0; --i) {

    next_multiplier = multiplier * size[i];

    if (relevant[i]) {
      temp = flat_state % next_multiplier;
      result[i] = temp/multiplier;
      flat_state -= temp;
      multiplier = next_multiplier;
    }
  }
}

bool IsStrictSubsetOf(const vector<bool>& first, const vector<bool>& second) {
  assert (first.size() == second.size());
  bool strict_larger = false;
  for (unsigned int j = 0; j < first.size(); ++j) {
    if ((first[j]) && (!second[j]))
      return false;
    if ((!first[j]) && (second[j]))
      strict_larger = true;
  }
  return strict_larger;
}

vector<string> explode(string const & s, char delim)
{
    vector<string> result;
    istringstream iss(s);

    for (string token; getline(iss, token, delim); )
    {
        result.push_back(move(token));
    }

    return result;
}
