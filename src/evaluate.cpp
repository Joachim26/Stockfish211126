/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include <random>
#include <chrono>

#include "incbin/incbin.h"
#include "misc.h"
#include "nnue/evaluate_nnue.h"
#include "nnue/nnue_architecture.h"
#include "position.h"
#include "types.h"
#include "uci.h"

namespace Stockfish {

namespace Eval {

int NNUE::RandomEval = 0;
int NNUE::WaitMs = 0;

// Tries to load a NNUE network at startup time, or when the engine
// receives a UCI command "setoption name EvalFile value nn-[a-z0-9]{12}.nnue"
// The name of the NNUE network is always retrieved from the EvalFile option.
// We search the given network in three locations: internally (the default
// network may be embedded in the binary), in the active working directory and
// in the engine directory. Distro packagers may define the DEFAULT_NNUE_DIRECTORY
// variable to have the engine search in a special directory in their distro.
NNUE::EvalFiles NNUE::load_networks(const std::string& rootDirectory,
                                    const OptionsMap&  options,
                                    NNUE::EvalFiles    evalFiles) {

    for (auto& [netSize, evalFile] : evalFiles)
    {
        std::string user_eval_file = options[evalFile.optionName];

        if (user_eval_file.empty())
            user_eval_file = evalFile.defaultName;

#if defined(DEFAULT_NNUE_DIRECTORY)
        std::vector<std::string> dirs = {"<internal>", "", rootDirectory,
                                         stringify(DEFAULT_NNUE_DIRECTORY)};
#else
        std::vector<std::string> dirs = {"<internal>", "", rootDirectory};
#endif

        for (const std::string& directory : dirs)
        {
            if (evalFile.current != user_eval_file)
            {
                if (directory != "<internal>")
                {
                    std::ifstream stream(directory + user_eval_file, std::ios::binary);
                    auto          description = NNUE::load_eval(stream, netSize);

                    if (description.has_value())
                    {
                        evalFile.current        = user_eval_file;
                        evalFile.netDescription = description.value();
                    }
                }

                if (directory == "<internal>" && user_eval_file == evalFile.defaultName)
                {
                    // C++ way to prepare a buffer for a memory stream
                    class MemoryBuffer: public std::basic_streambuf<char> {
                       public:
                        MemoryBuffer(char* p, size_t n) {
                            setg(p, p, p + n);
                            setp(p, p + n);
                        }
                    };

                    MemoryBuffer buffer(
                      const_cast<char*>(reinterpret_cast<const char*>(
                        netSize == Small ? gEmbeddedNNUESmallData : gEmbeddedNNUEBigData)),
                      size_t(netSize == Small ? gEmbeddedNNUESmallSize : gEmbeddedNNUEBigSize));
                    (void) gEmbeddedNNUEBigEnd;  // Silence warning on unused variable
                    (void) gEmbeddedNNUESmallEnd;

                    std::istream stream(&buffer);
                    auto         description = NNUE::load_eval(stream, netSize);

                    if (description.has_value())
                    {
                        evalFile.current        = user_eval_file;
                        evalFile.netDescription = description.value();
                    }
                }
            }
        }
    }

    return evalFiles;
}

// Verifies that the last net used was loaded successfully
void NNUE::verify(const OptionsMap&                                        options,
                  const std::unordered_map<Eval::NNUE::NetSize, EvalFile>& evalFiles) {

    for (const auto& [netSize, evalFile] : evalFiles)
    {
        std::string user_eval_file = options[evalFile.optionName];

        if (user_eval_file.empty())
            user_eval_file = evalFile.defaultName;

        if (evalFile.current != user_eval_file)
        {
            std::string msg1 =
              "Network evaluation parameters compatible with the engine must be available.";
            std::string msg2 =
              "The network file " + user_eval_file + " was not loaded successfully.";
            std::string msg3 = "The UCI option EvalFile might need to specify the full path, "
                               "including the directory name, to the network file.";
            std::string msg4 = "The default net can be downloaded from: "
                               "https://tests.stockfishchess.org/api/nn/"
                             + evalFile.defaultName;
            std::string msg5 = "The engine will be terminated now.";

            sync_cout << "info string ERROR: " << msg1 << sync_endl;
            sync_cout << "info string ERROR: " << msg2 << sync_endl;
            sync_cout << "info string ERROR: " << msg3 << sync_endl;
            sync_cout << "info string ERROR: " << msg4 << sync_endl;
            sync_cout << "info string ERROR: " << msg5 << sync_endl;

            exit(EXIT_FAILURE);
        }

        sync_cout << "info string NNUE evaluation using " << user_eval_file << sync_endl;
    }
}
}

// Returns a static, purely materialistic evaluation of the position from
// the point of view of the given color. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position& pos, Color c) {
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
         + (pos.non_pawn_material(c) - pos.non_pawn_material(~c));
}


// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks& networks, const Position& pos, int optimism) {

    assert(!pos.checkers());

    int  simpleEval = simple_eval(pos, pos.side_to_move());
    bool smallNet   = std::abs(simpleEval) > SmallNetThreshold;
    bool psqtOnly   = std::abs(simpleEval) > PsqtOnlyThreshold;

    int nnueComplexity;

    Value nnue = smallNet ? networks.small.evaluate(pos, true, &nnueComplexity, psqtOnly)
                          : networks.big.evaluate(pos, true, &nnueComplexity, false);

    // Blend optimism and eval with nnue complexity and material imbalance
    optimism += optimism * (nnueComplexity + std::abs(simpleEval - nnue)) / 524;
    nnue -= nnue * (nnueComplexity + std::abs(simpleEval - nnue)) / 31950;

    int npm = pos.non_pawn_material() / 64;
    int v   = (nnue * (927 + npm + 9 * pos.count<PAWN>()) + optimism * (159 + npm)) / 1000;

    // Damp down the evaluation linearly when shuffling
    int shuffling = pos.rule50_count();
    v             = v * (195 - shuffling) / 228;

    // SFnps Begin //
    if((NNUE::RandomEval) || (NNUE::WaitMs))
    {
      // waitms millisecs
      std::this_thread::sleep_for(std::chrono::milliseconds(NNUE::WaitMs));

      // RandomEval
      static thread_local std::mt19937_64 rng = [](){return std::mt19937_64(std::time(0));}();
      std::normal_distribution<float> d(0.0, PawnValue);
      float r = d(rng);
      r = std::clamp<float>(r, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
      v = (NNUE::RandomEval * Value(r) + (100 - NNUE::RandomEval) * v) / 100;
    }
    // SFnps End //




    // Guarantee evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Networks& networks) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, networks) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    Value v = networks.big.evaluate(pos, false);
    v       = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCI::to_cp(v) << " (white side)\n";

    v = evaluate(networks, pos, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "Final evaluation       " << 0.01 * UCI::to_cp(v) << " (white side)";
    ss << " [with scaled NNUE, ...]";
    ss << "\n";

    return ss.str();
}

}  // namespace Stockfish
