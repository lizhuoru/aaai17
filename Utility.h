#include <vector>
#include <cfloat>   // for FLT_MAX
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <utility>

using namespace std;

/* Returns a random integer in [0,max] */
int randInRange(int max);
int CheckAndMapParentFSA(const vector<int>& current_state,
    const vector<int>& next_state, const vector<int>& feature_size,
    const vector<bool>& parent_features);
int MapFactoredStateToInt(const vector<int>& state,
    const vector<int>& size, const vector<bool>& relevant);
void MapIntStateToVector(int flat_state,
    const vector<int>& size, const vector<bool>& relevant, vector<int>& result);
bool IsStrictSubsetOf(const vector<bool>& first, const vector<bool>& second);
vector<string> explode(string const & s, char delim);
