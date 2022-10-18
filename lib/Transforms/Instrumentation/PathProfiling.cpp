//===- PathProfiling.cpp - Inserts counters for path profiling ------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass instruments functions for Ball-Larus path profiling.  Ball-Larus
// profiling converts the CFG into a DAG by replacing backedges with edges
// from entry to the start block and from the end block to exit.  The paths
// along the new DAG are enumrated, i.e. each path is given a path number.
// Edges are instrumented to increment the path number register, such that the
// path number register will equal the path number of the path taken at the
// exit.
//
// This file defines classes for building a CFG for use with different stages
// in the Ball-Larus path profiling instrumentation [Ball96].  The
// requirements are formatting the llvm CFG into the Ball-Larus DAG, path
// numbering, finding a spanning tree, moving increments from the spanning
// tree to chords.
//
// Terms:
// DAG            - Directed Acyclic Graph.
// Ball-Larus DAG - A CFG with an entry node, an exit node, and backedges
//                  removed in the following manner.  For every backedge
//                  v->w, insert edge ENTRY->w and edge v->EXIT.
// Path Number    - The number corresponding to a specific path through a
//                  Ball-Larus DAG.
// Spanning Tree  - A subgraph, S, is a spanning tree if S covers all
//                  vertices and is a tree.
// Chord          - An edge not in the spanning tree.
//
// [Ball96]
//  T. Ball and J. R. Larus. "Efficient Path Profiling."
//  International Symposium on Microarchitecture, pages 46-57, 1996.
//  http://portal.acm.org/citation.cfm?id=243857
//
// [Ball94]
//  Thomas Ball.  "Efficiently Counting Program Events with Support for
//  On-line queries."
//  ACM Transactions on Programmmg Languages and Systems, Vol 16, No 5,
//  September 1994, Pages 1399-1410.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "path-profiling"

#include <llvm/ADT/Triple.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <vector>

// AFL++ headers
#include "debug.h"

#include "Analysis/PathNumbering.h"
#include "PathProfiling.h"

using namespace llvm;

namespace {
class BLInstrumentationNode;
class BLInstrumentationEdge;
class BLInstrumentationDag;

// ---------------------------------------------------------------------------
// BLInstrumentationNode extends BallLarusNode with member used by the
// instrumentation algortihms.
// ---------------------------------------------------------------------------
class BLInstrumentationNode : public BallLarusNode {
public:
  // Creates a new BLInstrumentationNode from a BasicBlock.
  BLInstrumentationNode(BasicBlock *BB);

  // Get/sets the Value corresponding to the pathNumber register,
  // constant or phinode.  Used by the instrumentation code to remember
  // path number Values.
  Value *getStartingPathNumber();
  void setStartingPathNumber(Value *pathNumber);

  Value *getEndingPathNumber();
  void setEndingPathNumber(Value *pathNumber);

  // Get/set the PHINode Instruction for this node.
  PHINode *getPathPHI();
  void setPathPHI(PHINode *pathPHI);

private:
  Value *_startingPathNumber; // The Value for the current pathNumber.
  Value *_endingPathNumber;   // The Value for the current pathNumber.
  PHINode *_pathPHI;          // The PHINode for current pathNumber.
};

// --------------------------------------------------------------------------
// BLInstrumentationEdge extends BallLarusEdge with data about the
// instrumentation that will end up on each edge.
// --------------------------------------------------------------------------
class BLInstrumentationEdge : public BallLarusEdge {
public:
  BLInstrumentationEdge(BLInstrumentationNode *source,
                        BLInstrumentationNode *target);

  // Sets the target node of this edge.  Required to split edges.
  void setTarget(BallLarusNode *node);

  // Get/set whether edge is in the spanning tree.
  bool isInSpanningTree() const;
  void setIsInSpanningTree(bool isInSpanningTree);

  // Get/ set whether this edge will be instrumented with a path number
  // initialization.
  bool isInitialization() const;
  void setIsInitialization(bool isInitialization);

  // Get/set whether this edge will be instrumented with a path counter
  // increment.  Notice this is incrementing the path counter
  // corresponding to the path number register.  The path number
  // increment is determined by getIncrement().
  bool isCounterIncrement() const;
  void setIsCounterIncrement(bool isCounterIncrement);

  // Get/set the path number increment that this edge will be instrumented
  // with.  This is distinct from the path counter increment and the
  // weight.  The counter increment counts the number of executions of
  // some path, whereas the path number keeps track of which path number
  // the program is on.
  long getIncrement() const;
  void setIncrement(long increment);

  // Get/set whether the edge has been instrumented.
  bool hasInstrumentation();
  void setHasInstrumentation(bool hasInstrumentation);

  // Returns the successor number of this edge in the source.
  unsigned getSuccessorNumber();

private:
  // The increment that the code will be instrumented with.
  long long _increment;

  // Whether this edge is in the spanning tree.
  bool _isInSpanningTree;

  // Whether this edge is an initialiation of the path number.
  bool _isInitialization;

  // Whether this edge is a path counter increment.
  bool _isCounterIncrement;

  // Whether this edge has been instrumented.
  bool _hasInstrumentation;
};

// ---------------------------------------------------------------------------
// BLInstrumentationDag extends BallLarusDag with algorithms that
// determine where instrumentation should be placed.
// ---------------------------------------------------------------------------
class BLInstrumentationDag : public BallLarusDag {
public:
  BLInstrumentationDag(Function &F);

  // Returns the Exit->Root edge. This edge is required for creating
  // directed cycles in the algorithm for moving instrumentation off of
  // the spanning tree
  BallLarusEdge *getExitRootEdge();

  // Returns an array of phony edges which mark those nodes
  // with function calls
  BLEdgeVector getCallPhonyEdges();

  // Gets/sets the path counter array
  Constant *getCounterArray();
  void setCounterArray(Constant *c);

  // Calculates the increments for the chords, thereby removing
  // instrumentation from the spanning tree edges. Implementation is based
  // on the algorithm in Figure 4 of [Ball94]
  void calculateChordIncrements();

  // Updates the state when an edge has been split
  void splitUpdate(BLInstrumentationEdge *formerEdge, BasicBlock *newBlock);

  // Calculates a spanning tree of the DAG ignoring cycles.  Whichever
  // edges are in the spanning tree will not be instrumented, but this
  // implementation does not try to minimize the instrumentation overhead
  // by trying to find hot edges.
  void calculateSpanningTree();

  // Pushes initialization further down in order to group the first
  // increment and initialization.
  void pushInitialization();

  // Pushes the path counter increments up in order to group the last path
  // number increment.
  void pushCounters();

  // Removes phony edges from the successor list of the source, and the
  // predecessor list of the target.
  void unlinkPhony();

  // Generate dot graph for the function
  void generateDotGraph();

protected:
  // BLInstrumentationDag creates BLInstrumentationNode objects in this
  // method overriding the creation of BallLarusNode objects.
  //
  // Allows subclasses to determine which type of Node is created.
  // Override this method to produce subclasses of BallLarusNode if
  // necessary.
  virtual BallLarusNode *createNode(BasicBlock *BB);

  // BLInstrumentationDag create BLInstrumentationEdges.
  //
  // Allows subclasses to determine which type of Edge is created.
  // Override this method to produce subclasses of BallLarusEdge if
  // necessary.  Parameters source and target will have been created by
  // createNode and can be cast to the subclass of BallLarusNode*
  // returned by createNode.
  virtual BallLarusEdge *createEdge(BallLarusNode *source,
                                    BallLarusNode *target, unsigned edgeNumber);

private:
  BLEdgeVector _treeEdges;  // All edges in the spanning tree.
  BLEdgeVector _chordEdges; // All edges not in the spanning tree.
  Constant *_counterArray;  // Array to store path counters

  // Removes the edge from the appropriate predecessor and successor lists.
  void unlinkEdge(BallLarusEdge *edge);

  // Makes an edge part of the spanning tree.
  void makeEdgeSpanning(BLInstrumentationEdge *edge);

  // Pushes initialization and calls itself recursively.
  void pushInitializationFromEdge(BLInstrumentationEdge *edge);

  // Pushes path counter increments up recursively.
  void pushCountersFromEdge(BLInstrumentationEdge *edge);

  // Depth first algorithm for determining the chord increments.f
  void calculateChordIncrementsDfs(long weight, BallLarusNode *v,
                                   BallLarusEdge *e);

  // Determines the relative direction of two edges.
  int calculateChordIncrementsDir(BallLarusEdge *e, BallLarusEdge *f);
};

// ---------------------------------------------------------------------------
// PathProfiler is a module pass which instruments path profiling instructions
// ---------------------------------------------------------------------------
class PathProfiler : public ModulePass {
private:
  // Current context for multi threading support.
  LLVMContext *Context;

  // Data layout
  const DataLayout *DL;

  // The target triple
  Triple TargetTriple;

  // Unique identifier for the module being instrumented
  std::string CurModuleUniqueId;

  SmallVector<GlobalValue *, 20> GlobalsToAppendToUsed;
  SmallVector<GlobalValue *, 20> GlobalsToAppendToCompilerUsed;

  // Useful types
  Type *VoidPtrTy;
  Type *Int8Ty;
  Type *Int32Ty;

  // The type of an entry in the function table
  StructType *FTEntryTy;

  // Which function are we currently instrumenting
  unsigned currentFunctionNumber;

  // Number of currently-instrumented functions
  unsigned numInstrumentedFunctions;

  // For debugging
  FunctionCallee debugFunction;

  // Instruments each function with path profiling.  'main' is instrumented
  // with code to save the profile to disk.
  bool runOnModule(Module &M) override;

  // Analyzes the function for Ball-Larus path profiling, and inserts code.
  Constant *runOnFunction(Function &F, Module &M);

  // Creates an increment constant representing incr.
  ConstantInt *createIncrementConstant(long incr, int bitsize);

  // Creates an increment constant representing the value in
  // edge->getIncrement().
  ConstantInt *createIncrementConstant(BLInstrumentationEdge *edge);

  // Finds the insertion point after pathNumber in block.  PathNumber may
  // be NULL.
  BasicBlock::iterator getInsertionPoint(BasicBlock *block, Value *pathNumber);

  // Inserts source's pathNumber Value* into target.  Target may or may not
  // have multiple predecessors, and may or may not have its phiNode
  // initalized.
  void pushValueIntoNode(BLInstrumentationNode *source,
                         BLInstrumentationNode *target);

  // Inserts source's pathNumber Value* into the appropriate slot of
  // target's phiNode.
  void pushValueIntoPHI(BLInstrumentationNode *target,
                        BLInstrumentationNode *source);

  // The Value* in node, oldVal,  is updated with a Value* correspodning to
  // oldVal + addition.
  void insertNumberIncrement(BLInstrumentationNode *node, Value *addition,
                             bool atBeginning);

  // Creates a counter increment in the given node.  The Value* in node is
  // taken as the index into a hash table.
  void insertCounterIncrement(Value *incValue, Instruction *insertPoint,
                              BLInstrumentationDag *dag, Constant *FTEntry,
                              bool increment = true);

  // A PHINode is created in the node, and its values initialized to -1U.
  void preparePHI(BLInstrumentationNode *node);

  // Inserts instrumentation for the given edge
  //
  // Pre: The edge's source node has pathNumber set if edge is non zero
  // path number increment.
  //
  // Post: Edge's target node has a pathNumber set to the path number Value
  // corresponding to the value of the path register after edge's
  // execution.
  void insertInstrumentationStartingAt(BLInstrumentationEdge *edge,
                                       BLInstrumentationDag *dag,
                                       Constant *FTEntry);

  // If this edge is a critical edge, then inserts a node at this edge.
  // This edge becomes the first edge, and a new BallLarusEdge is created.
  bool splitCritical(BLInstrumentationEdge *edge, BLInstrumentationDag *dag);

  // Inserts instrumentation according to the marked edges in dag.  Phony
  // edges must be unlinked from the DAG, but accessible from the
  // backedges.  Dag must have initializations, path number increments, and
  // counter increments present.
  //
  // Counter storage is created here.
  void insertInstrumentation(BLInstrumentationDag &dag, Constant *FTEntry,
                             Module &M);

  std::string getSectionStart(const std::string &Section) const;
  std::string getSectionEnd(const std::string &Section) const;
  std::string getSectionName(const std::string &Section) const;

  std::pair<Value *, Value *> CreateSecStartEnd(Module &M, const char *Section,
                                                Type *Ty);

  // Insert a constructor that records the number of paths in this module
  Function *createInitCallForSection(Module &M);

public:
  static char ID; // Pass identification, replacement for typeid
  PathProfiler() : ModulePass(ID) {}

  virtual StringRef getPassName() const override { return "Path Profiler"; }
};

static constexpr const uint64_t kPathProfCtorPriority = 2;
static constexpr const char kPathProfCtorName[] = "path_profiler.ctor";
static constexpr const char kPathProfInitFunctionName[] =
    "__path_profiler_init";
static constexpr const char kPathProfSectionName[] = "path_profilers";

// Should we print the dot-graphs
static cl::opt<bool>
    DotPathDag("path-profile-pathdag", cl::Hidden,
               cl::desc("Output the path profiling DAG for each function."));

static cl::opt<uint64_t>
    HashThreshold("path-profile-hash-threshold", cl::Hidden,
                  cl::desc("Set the hash table threshold."), cl::init(1000));
} // end anonymous namespace

// Register the path profiler as a pass
char PathProfiler::ID = 0;

namespace llvm {
// BallLarusEdge << operator overloading
raw_ostream &operator<<(raw_ostream &os,
                        const BLInstrumentationEdge &edge) LLVM_ATTRIBUTE_USED;
raw_ostream &operator<<(raw_ostream &os, const BLInstrumentationEdge &edge) {
  os << "[" << edge.getSource()->getName() << " -> "
     << edge.getTarget()->getName()
     << "] init: " << (edge.isInitialization() ? "yes" : "no")
     << " incr:" << edge.getIncrement()
     << " cinc: " << (edge.isCounterIncrement() ? "yes" : "no");
  return os;
}
} // namespace llvm

// Creates a new BLInstrumentationNode from a BasicBlock.
BLInstrumentationNode::BLInstrumentationNode(BasicBlock *BB)
    : BallLarusNode(BB), _startingPathNumber(nullptr),
      _endingPathNumber(nullptr), _pathPHI(nullptr) {}

// Constructor for BLInstrumentationEdge.
BLInstrumentationEdge::BLInstrumentationEdge(BLInstrumentationNode *source,
                                             BLInstrumentationNode *target)
    : BallLarusEdge(source, target, 0), _increment(0), _isInSpanningTree(false),
      _isInitialization(false), _isCounterIncrement(false),
      _hasInstrumentation(false) {}

// Sets the target node of this edge.  Required to split edges.
void BLInstrumentationEdge::setTarget(BallLarusNode *node) { _target = node; }

// Returns whether this edge is in the spanning tree.
bool BLInstrumentationEdge::isInSpanningTree() const {
  return _isInSpanningTree;
}

// Sets whether this edge is in the spanning tree.
void BLInstrumentationEdge::setIsInSpanningTree(bool isInSpanningTree) {
  _isInSpanningTree = isInSpanningTree;
}

// Returns whether this edge will be instrumented with a path number
// initialization.
bool BLInstrumentationEdge::isInitialization() const {
  return _isInitialization;
}

// Sets whether this edge will be instrumented with a path number
// initialization.
void BLInstrumentationEdge::setIsInitialization(bool isInitialization) {
  _isInitialization = isInitialization;
}

// Returns whether this edge will be instrumented with a path counter
// increment.  Notice this is incrementing the path counter
// corresponding to the path number register.  The path number
// increment is determined by getIncrement().
bool BLInstrumentationEdge::isCounterIncrement() const {
  return _isCounterIncrement;
}

// Sets whether this edge will be instrumented with a path counter
// increment.
void BLInstrumentationEdge::setIsCounterIncrement(bool isCounterIncrement) {
  _isCounterIncrement = isCounterIncrement;
}

// Gets the path number increment that this edge will be instrumented
// with.  This is distinct from the path counter increment and the
// weight.  The counter increment is counts the number of executions of
// some path, whereas the path number keeps track of which path number
// the program is on.
long BLInstrumentationEdge::getIncrement() const { return _increment; }

// Set whether this edge will be instrumented with a path number
// increment.
void BLInstrumentationEdge::setIncrement(long increment) {
  _increment = increment;
}

// True iff the edge has already been instrumented.
bool BLInstrumentationEdge::hasInstrumentation() { return _hasInstrumentation; }

// Set whether this edge has been instrumented.
void BLInstrumentationEdge::setHasInstrumentation(bool hasInstrumentation) {
  _hasInstrumentation = hasInstrumentation;
}

// Returns the successor number of this edge in the source.
unsigned BLInstrumentationEdge::getSuccessorNumber() {
  BallLarusNode *sourceNode = getSource();
  BallLarusNode *targetNode = getTarget();
  BasicBlock *source = sourceNode->getBlock();
  BasicBlock *target = targetNode->getBlock();

  if (source == nullptr || target == nullptr)
    return 0;

  auto *terminator = source->getTerminator();

  unsigned i;
  for (i = 0; i < terminator->getNumSuccessors(); i++) {
    if (terminator->getSuccessor(i) == target)
      break;
  }

  return i;
}

// BLInstrumentationDag constructor initializes a DAG for the given Function.
BLInstrumentationDag::BLInstrumentationDag(Function &F)
    : BallLarusDag(F), _counterArray(0) {}

// Returns the Exit->Root edge. This edge is required for creating
// directed cycles in the algorithm for moving instrumentation off of
// the spanning tree
BallLarusEdge *BLInstrumentationDag::getExitRootEdge() {
  BLEdgeIterator erEdge = getExit()->succBegin();
  return *erEdge;
}

BLEdgeVector BLInstrumentationDag::getCallPhonyEdges() {
  BLEdgeVector callEdges;

  for (BLEdgeIterator edge = _edges.begin(), end = _edges.end(); edge != end;
       edge++) {
    if ((*edge)->getType() == BallLarusEdge::CALLEDGE_PHONY)
      callEdges.push_back(*edge);
  }

  return callEdges;
}

// Gets the path counter array
Constant *BLInstrumentationDag::getCounterArray() { return _counterArray; }

void BLInstrumentationDag::setCounterArray(Constant *c) { _counterArray = c; }

// Calculates the increment for the chords, thereby removing
// instrumentation from the spanning tree edges. Implementation is based on
// the algorithm in Figure 4 of [Ball94]
void BLInstrumentationDag::calculateChordIncrements() {
  calculateChordIncrementsDfs(0, getRoot(), nullptr);

  BLInstrumentationEdge *chord;
  for (BLEdgeIterator chordEdge = _chordEdges.begin(), end = _chordEdges.end();
       chordEdge != end; chordEdge++) {
    chord = (BLInstrumentationEdge *)*chordEdge;
    chord->setIncrement(chord->getIncrement() + chord->getWeight());
  }
}

// Updates the state when an edge has been split
void BLInstrumentationDag::splitUpdate(BLInstrumentationEdge *formerEdge,
                                       BasicBlock *newBlock) {
  BallLarusNode *oldTarget = formerEdge->getTarget();
  BallLarusNode *newNode = addNode(newBlock);
  formerEdge->setTarget(newNode);
  newNode->addPredEdge(formerEdge);

  LLVM_DEBUG(dbgs() << "  Edge split: " << *formerEdge << "\n");

  oldTarget->removePredEdge(formerEdge);
  BallLarusEdge *newEdge = addEdge(newNode, oldTarget, 0);

  if (formerEdge->getType() == BallLarusEdge::BACKEDGE ||
      formerEdge->getType() == BallLarusEdge::SPLITEDGE) {
    newEdge->setType(formerEdge->getType());
    newEdge->setPhonyRoot(formerEdge->getPhonyRoot());
    newEdge->setPhonyExit(formerEdge->getPhonyExit());
    formerEdge->setType(BallLarusEdge::NORMAL);
    formerEdge->setPhonyRoot(nullptr);
    formerEdge->setPhonyExit(nullptr);
  }
}

// Calculates a spanning tree of the DAG ignoring cycles.  Whichever
// edges are in the spanning tree will not be instrumented, but this
// implementation does not try to minimize the instrumentation overhead
// by trying to find hot edges.
void BLInstrumentationDag::calculateSpanningTree() {
  std::stack<BallLarusNode *> dfsStack;

  for (BLNodeIterator nodeIt = _nodes.begin(), end = _nodes.end();
       nodeIt != end; nodeIt++) {
    (*nodeIt)->setColor(BallLarusNode::WHITE);
  }

  dfsStack.push(getRoot());
  while (dfsStack.size() > 0) {
    BallLarusNode *node = dfsStack.top();
    dfsStack.pop();

    if (node->getColor() == BallLarusNode::WHITE)
      continue;

    BallLarusNode *nextNode;
    bool forward = true;
    BLEdgeIterator succEnd = node->succEnd();

    node->setColor(BallLarusNode::WHITE);
    // first iterate over successors then predecessors
    for (BLEdgeIterator edge = node->succBegin(), predEnd = node->predEnd();
         edge != predEnd; edge++) {
      if (edge == succEnd) {
        edge = node->predBegin();
        forward = false;
      }

      // Ignore split edges
      if ((*edge)->getType() == BallLarusEdge::SPLITEDGE)
        continue;

      nextNode = forward ? (*edge)->getTarget() : (*edge)->getSource();
      if (nextNode->getColor() != BallLarusNode::WHITE) {
        nextNode->setColor(BallLarusNode::WHITE);
        makeEdgeSpanning((BLInstrumentationEdge *)(*edge));
      }
    }
  }

  for (BLEdgeIterator edge = _edges.begin(), end = _edges.end(); edge != end;
       edge++) {
    BLInstrumentationEdge *instEdge = (BLInstrumentationEdge *)(*edge);
    // safe since createEdge is overriden
    if (!instEdge->isInSpanningTree() &&
        (*edge)->getType() != BallLarusEdge::SPLITEDGE)
      _chordEdges.push_back(instEdge);
  }
}

// Pushes initialization further down in order to group the first
// increment and initialization.
void BLInstrumentationDag::pushInitialization() {
  BLInstrumentationEdge *exitRootEdge =
      (BLInstrumentationEdge *)getExitRootEdge();
  exitRootEdge->setIsInitialization(true);
  pushInitializationFromEdge(exitRootEdge);
}

// Pushes the path counter increments up in order to group the last path
// number increment.
void BLInstrumentationDag::pushCounters() {
  BLInstrumentationEdge *exitRootEdge =
      (BLInstrumentationEdge *)getExitRootEdge();
  exitRootEdge->setIsCounterIncrement(true);
  pushCountersFromEdge(exitRootEdge);
}

// Removes phony edges from the successor list of the source, and the
// predecessor list of the target.
void BLInstrumentationDag::unlinkPhony() {
  BallLarusEdge *edge;

  for (BLEdgeIterator next = _edges.begin(), end = _edges.end(); next != end;
       next++) {
    edge = (*next);

    if (edge->getType() == BallLarusEdge::BACKEDGE_PHONY ||
        edge->getType() == BallLarusEdge::SPLITEDGE_PHONY ||
        edge->getType() == BallLarusEdge::CALLEDGE_PHONY) {
      unlinkEdge(edge);
    }
  }
}

// Generate a .dot graph to represent the DAG and pathNumbers
void BLInstrumentationDag::generateDotGraph() {
  std::string functionName = getFunction().getName().str();
  std::string filename = "pathdag." + functionName + ".dot";

  LLVM_DEBUG(dbgs() << "Writing '" << filename << "'...\n");
  std::error_code EC;
  raw_fd_ostream dotFile(filename, EC);

  if (EC) {
    errs() << "Error opening '" << filename.c_str() << "' for writing!";
    errs() << "\n";
    return;
  }

  dotFile << "digraph " << functionName << " {\n";

  for (BLEdgeIterator edge = _edges.begin(), end = _edges.end(); edge != end;
       edge++) {
    std::string sourceName = (*edge)->getSource()->getName();
    std::string targetName = (*edge)->getTarget()->getName();

    dotFile << "\t\"" << sourceName.c_str() << "\" -> \"" << targetName.c_str()
            << "\" ";

    long inc = ((BLInstrumentationEdge *)(*edge))->getIncrement();

    switch ((*edge)->getType()) {
    case BallLarusEdge::NORMAL:
      dotFile << "[label=" << inc << "] [color=black];\n";
      break;

    case BallLarusEdge::BACKEDGE:
      dotFile << "[color=cyan];\n";
      break;

    case BallLarusEdge::BACKEDGE_PHONY:
      dotFile << "[label=" << inc << "] [color=blue];\n";
      break;

    case BallLarusEdge::SPLITEDGE:
      dotFile << "[color=violet];\n";
      break;

    case BallLarusEdge::SPLITEDGE_PHONY:
      dotFile << "[label=" << inc << "] [color=red];\n";
      break;

    case BallLarusEdge::CALLEDGE_PHONY:
      dotFile << "[label=" << inc << "] [color=green];\n";
      break;
    }
  }

  dotFile << "}\n";
}

// Allows subclasses to determine which type of Node is created.
// Override this method to produce subclasses of BallLarusNode if
// necessary. The destructor of BallLarusDag will call free on each pointer
// created.
BallLarusNode *BLInstrumentationDag::createNode(BasicBlock *BB) {
  return new BLInstrumentationNode(BB);
}

// Allows subclasses to determine which type of Edge is created.
// Override this method to produce subclasses of BallLarusEdge if
// necessary. The destructor of BallLarusDag will call free on each pointer
// created.
BallLarusEdge *BLInstrumentationDag::createEdge(BallLarusNode *source,
                                                BallLarusNode *target,
                                                unsigned edgeNumber) {
  // One can cast from BallLarusNode to BLInstrumentationNode since createNode
  // is overriden to produce BLInstrumentationNode.
  return new BLInstrumentationEdge((BLInstrumentationNode *)source,
                                   (BLInstrumentationNode *)target);
}

// Sets the Value corresponding to the pathNumber register, constant,
// or phinode.  Used by the instrumentation code to remember path
// number Values.
Value *BLInstrumentationNode::getStartingPathNumber() {
  return _startingPathNumber;
}

// Sets the Value of the pathNumber.  Used by the instrumentation code.
void BLInstrumentationNode::setStartingPathNumber(Value *pathNumber) {
  LLVM_DEBUG(dbgs() << "  SPN-" << getName() << " <-- "
                    << (pathNumber ? pathNumber->getName() : "unused") << "\n");
  _startingPathNumber = pathNumber;
}

Value *BLInstrumentationNode::getEndingPathNumber() {
  return _endingPathNumber;
}

void BLInstrumentationNode::setEndingPathNumber(Value *pathNumber) {
  LLVM_DEBUG(dbgs() << "  EPN-" << getName() << " <-- "
                    << (pathNumber ? pathNumber->getName() : "unused") << "\n");
  _endingPathNumber = pathNumber;
}

// Get the PHINode Instruction for this node.  Used by instrumentation
// code.
PHINode *BLInstrumentationNode::getPathPHI() { return _pathPHI; }

// Set the PHINode Instruction for this node.  Used by instrumentation
// code.
void BLInstrumentationNode::setPathPHI(PHINode *pathPHI) { _pathPHI = pathPHI; }

// Removes the edge from the appropriate predecessor and successor
// lists.
void BLInstrumentationDag::unlinkEdge(BallLarusEdge *edge) {
  if (edge == getExitRootEdge())
    LLVM_DEBUG(dbgs() << " Removing exit->root edge\n");

  edge->getSource()->removeSuccEdge(edge);
  edge->getTarget()->removePredEdge(edge);
}

// Makes an edge part of the spanning tree.
void BLInstrumentationDag::makeEdgeSpanning(BLInstrumentationEdge *edge) {
  edge->setIsInSpanningTree(true);
  _treeEdges.push_back(edge);
}

// Pushes initialization and calls itself recursively.
void BLInstrumentationDag::pushInitializationFromEdge(
    BLInstrumentationEdge *edge) {
  BallLarusNode *target;

  target = edge->getTarget();
  if (target->getNumberPredEdges() > 1 || target == getExit()) {
    return;
  } else {
    for (BLEdgeIterator next = target->succBegin(), end = target->succEnd();
         next != end; next++) {
      BLInstrumentationEdge *intoEdge = (BLInstrumentationEdge *)*next;

      // Skip split edges
      if (intoEdge->getType() == BallLarusEdge::SPLITEDGE)
        continue;

      intoEdge->setIncrement(intoEdge->getIncrement() + edge->getIncrement());
      intoEdge->setIsInitialization(true);
      pushInitializationFromEdge(intoEdge);
    }

    edge->setIncrement(0);
    edge->setIsInitialization(false);
  }
}

// Pushes path counter increments up recursively.
void BLInstrumentationDag::pushCountersFromEdge(BLInstrumentationEdge *edge) {
  BallLarusNode *source;

  source = edge->getSource();
  if (source->getNumberSuccEdges() > 1 || source == getRoot() ||
      edge->isInitialization()) {
    return;
  } else {
    for (BLEdgeIterator previous = source->predBegin(), end = source->predEnd();
         previous != end; previous++) {
      BLInstrumentationEdge *fromEdge = (BLInstrumentationEdge *)*previous;

      // Skip split edges
      if (fromEdge->getType() == BallLarusEdge::SPLITEDGE)
        continue;

      fromEdge->setIncrement(fromEdge->getIncrement() + edge->getIncrement());
      fromEdge->setIsCounterIncrement(true);
      pushCountersFromEdge(fromEdge);
    }

    edge->setIncrement(0);
    edge->setIsCounterIncrement(false);
  }
}

// Depth first algorithm for determining the chord increments.
void BLInstrumentationDag::calculateChordIncrementsDfs(long weight,
                                                       BallLarusNode *v,
                                                       BallLarusEdge *e) {
  BLInstrumentationEdge *f;

  for (BLEdgeIterator treeEdge = _treeEdges.begin(), end = _treeEdges.end();
       treeEdge != end; treeEdge++) {
    f = (BLInstrumentationEdge *)*treeEdge;
    if (e != f && v == f->getTarget()) {
      calculateChordIncrementsDfs(calculateChordIncrementsDir(e, f) * (weight) +
                                      f->getWeight(),
                                  f->getSource(), f);
    }
    if (e != f && v == f->getSource()) {
      calculateChordIncrementsDfs(calculateChordIncrementsDir(e, f) * (weight) +
                                      f->getWeight(),
                                  f->getTarget(), f);
    }
  }

  for (BLEdgeIterator chordEdge = _chordEdges.begin(), end = _chordEdges.end();
       chordEdge != end; chordEdge++) {
    f = (BLInstrumentationEdge *)*chordEdge;
    if (v == f->getSource() || v == f->getTarget()) {
      f->setIncrement(f->getIncrement() +
                      calculateChordIncrementsDir(e, f) * weight);
    }
  }
}

// Determines the relative direction of two edges.
int BLInstrumentationDag::calculateChordIncrementsDir(BallLarusEdge *e,
                                                      BallLarusEdge *f) {
  if (e == nullptr)
    return 1;
  else if (e->getSource() == f->getTarget() || e->getTarget() == f->getSource())
    return 1;

  return -1;
}

// Creates an increment constant representing incr.
ConstantInt *PathProfiler::createIncrementConstant(long incr, int bitsize) {
  return ConstantInt::get(IntegerType::get(*Context, bitsize), incr);
}

// Creates an increment constant representing the value in
// edge->getIncrement().
ConstantInt *
PathProfiler::createIncrementConstant(BLInstrumentationEdge *edge) {
  return createIncrementConstant(edge->getIncrement(), 32);
}

// Finds the insertion point after pathNumber in block.  PathNumber may
// be NULL.
BasicBlock::iterator PathProfiler::getInsertionPoint(BasicBlock *block,
                                                     Value *pathNumber) {
  if (pathNumber == nullptr || isa<ConstantInt>(pathNumber) ||
      (((Instruction *)(pathNumber))->getParent()) != block) {
    return block->getFirstInsertionPt();
  } else {
    Instruction *pathNumberInst = (Instruction *)(pathNumber);
    BasicBlock::iterator insertPoint;
    BasicBlock::iterator end = block->end();

    for (insertPoint = block->begin(); insertPoint != end; insertPoint++) {
      Instruction *insertInst = &(*insertPoint);

      if (insertInst == pathNumberInst)
        return ++insertPoint;
    }

    return insertPoint;
  }
}

// A PHINode is created in the node, and its values initialized to -1U.
void PathProfiler::preparePHI(BLInstrumentationNode *node) {
  BasicBlock *block = node->getBlock();
  auto insertPoint = block->getFirstInsertionPt();
  pred_iterator PB = pred_begin(node->getBlock()),
                PE = pred_end(node->getBlock());
  PHINode *phi = PHINode::Create(Int32Ty, std::distance(PB, PE), "pathNumber",
                                 &*insertPoint);
  node->setPathPHI(phi);
  node->setStartingPathNumber(phi);
  node->setEndingPathNumber(phi);

  for (pred_iterator predIt = PB; predIt != PE; predIt++) {
    BasicBlock *pred = (*predIt);

    if (pred != nullptr)
      phi->addIncoming(createIncrementConstant((long)-1, 32), pred);
  }
}

// Inserts source's pathNumber Value* into target.  Target may or may not
// have multiple predecessors, and may or may not have its phiNode
// initalized.
void PathProfiler::pushValueIntoNode(BLInstrumentationNode *source,
                                     BLInstrumentationNode *target) {
  if (target->getBlock() == nullptr)
    return;

  if (target->getNumberPredEdges() <= 1) {
    assert(target->getStartingPathNumber() == nullptr &&
           "Target already has path number");
    target->setStartingPathNumber(source->getEndingPathNumber());
    target->setEndingPathNumber(source->getEndingPathNumber());
    LLVM_DEBUG(dbgs() << "  Passing path number"
                      << (source->getEndingPathNumber() ? "" : " (null)")
                      << " value through.\n");
  } else {
    if (target->getPathPHI() == nullptr) {
      LLVM_DEBUG(dbgs() << "  Initializing PHI node for block '"
                        << target->getName() << "'\n");
      preparePHI(target);
    }
    pushValueIntoPHI(target, source);
    LLVM_DEBUG(dbgs() << "  Passing number value into PHI for block '"
                      << target->getName() << "'\n");
  }
}

// Inserts source's pathNumber Value* into the appropriate slot of
// target's phiNode.
void PathProfiler::pushValueIntoPHI(BLInstrumentationNode *target,
                                    BLInstrumentationNode *source) {
  PHINode *phi = target->getPathPHI();
  assert(phi != nullptr && "  Tried to push value into node with PHI, but node"
                           " actually had no PHI.");
  phi->removeIncomingValue(source->getBlock(), false);
  phi->addIncoming(source->getEndingPathNumber(), source->getBlock());
}

// The Value* in node, oldVal,  is updated with a Value* correspodning to
// oldVal + addition.
void PathProfiler::insertNumberIncrement(BLInstrumentationNode *node,
                                         Value *addition, bool atBeginning) {
  BasicBlock *block = node->getBlock();
  assert(node->getStartingPathNumber() != nullptr);
  assert(node->getEndingPathNumber() != nullptr);

  auto insertPoint = [&]() {
    if (atBeginning)
      return &*block->getFirstInsertionPt();
    else
      return block->getTerminator();
  }();

  LLVM_DEBUG(errs() << "  Creating addition instruction.\n");
  Value *newpn =
      BinaryOperator::Create(Instruction::Add, node->getStartingPathNumber(),
                             addition, "pathNumber", insertPoint);

  node->setEndingPathNumber(newpn);

  if (atBeginning)
    node->setStartingPathNumber(newpn);
}

// Creates a counter increment in the given node.  The Value* in node is
// taken as the index into an array or hash table.  The hash table access
// is a call to the runtime.
void PathProfiler::insertCounterIncrement(Value *incValue,
                                          Instruction *insertPoint,
                                          BLInstrumentationDag *dag,
                                          Constant *FTEntry, bool increment) {
  // XXX hack to handle calls to exit, abort, etc.
  if (isa<UnreachableInst>(insertPoint)) {
    insertPoint =
        insertPoint->getPrevNonDebugInstruction(/*SkipPseudoOp=*/true);
  }

  // Counter increment for array
  if (dag->getNumberOfPaths() <= HashThreshold) {
    CallInst::Create(debugFunction, {FTEntry, incValue}, "", insertPoint);

    // Get pointer to the "array" index
    GetElementPtrInst *pcPointer = GetElementPtrInst::CreateInBounds(
        Int8Ty,
        new LoadInst(VoidPtrTy, dag->getCounterArray(), "counters",
                     insertPoint),
        incValue, "counterInc", insertPoint);

    // Load from the array - call it oldPC
    LoadInst *oldPc = new LoadInst(Int8Ty, pcPointer, "oldPC", insertPoint);
    oldPc->setMetadata(oldPc->getModule()->getMDKindID("nosanitize"),
                       MDNode::get(*Context, None));

    // Test to see whether adding 1 will overflow the counter
    ICmpInst *isMax =
        new ICmpInst(insertPoint, CmpInst::ICMP_ULT, oldPc,
                     createIncrementConstant(UINT8_MAX, 8), "isMax");

    // Select increment for the path counter based on overflow
    SelectInst *inc = SelectInst::Create(
        isMax, createIncrementConstant(increment ? 1 : -1, 8),
        createIncrementConstant(0, 8), "pathInc", insertPoint);

    // newPc = oldPc + inc
    BinaryOperator *newPc = BinaryOperator::Create(Instruction::Add, oldPc, inc,
                                                   "newPC", insertPoint);

    // Store back in to the array
    auto *Store = new StoreInst(newPc, pcPointer, insertPoint);
    Store->setMetadata(Store->getModule()->getMDKindID("nosanitize"),
                       MDNode::get(*Context, None));
  }
}

// Inserts instrumentation for the given edge
//
// Pre: The edge's source node has pathNumber set if edge is non zero
// path number increment.
//
// Post: Edge's target node has a pathNumber set to the path number Value
// corresponding to the value of the path register after edge's
// execution.
//
// FIXME: This should be reworked so it's not recursive.
void PathProfiler::insertInstrumentationStartingAt(BLInstrumentationEdge *edge,
                                                   BLInstrumentationDag *dag,
                                                   Constant *FTEntry) {
  // Mark the edge as instrumented
  edge->setHasInstrumentation(true);
  LLVM_DEBUG(dbgs() << "\nInstrumenting edge: " << (*edge) << "\n");

  // create a new node for this edge's instrumentation
  splitCritical(edge, dag);

  BLInstrumentationNode *sourceNode =
      (BLInstrumentationNode *)edge->getSource();
  BLInstrumentationNode *targetNode =
      (BLInstrumentationNode *)edge->getTarget();
  BLInstrumentationNode *instrumentNode;
  BLInstrumentationNode *nextSourceNode;

  bool atBeginning = false;

  // Source node has only 1 successor so any information can be simply
  // inserted in to it without splitting
  if (sourceNode->getBlock() && sourceNode->getNumberSuccEdges() <= 1) {
    LLVM_DEBUG(dbgs() << "  Potential instructions to be placed in: "
                      << sourceNode->getName() << " (at end)\n");
    instrumentNode = sourceNode;
    nextSourceNode = targetNode; // ... since we never made any new nodes
  }

  // The target node only has one predecessor, so we can safely insert edge
  // instrumentation into it. If there was splitting, it must have been
  // successful.
  else if (targetNode->getNumberPredEdges() == 1) {
    LLVM_DEBUG(dbgs() << "  Potential instructions to be placed in: "
                      << targetNode->getName() << " (at beginning)\n");
    pushValueIntoNode(sourceNode, targetNode);
    instrumentNode = targetNode;
    nextSourceNode = nullptr; // ... otherwise we'll just keep splitting
    atBeginning = true;
  }

  // Somehow, splitting must have failed.
  else {
    errs() << "Instrumenting could not split a critical edge.\n";
    LLVM_DEBUG(dbgs() << "  Couldn't split edge " << (*edge) << ".\n");
    return;
  }

  // Insert instrumentation if this is a back or split edge
  if (edge->getType() == BallLarusEdge::BACKEDGE ||
      edge->getType() == BallLarusEdge::SPLITEDGE) {
    BLInstrumentationEdge *top = (BLInstrumentationEdge *)edge->getPhonyRoot();
    BLInstrumentationEdge *bottom =
        (BLInstrumentationEdge *)edge->getPhonyExit();

    assert(top->isInitialization() && " Top phony edge did not"
                                      " contain a path number initialization.");
    assert(bottom->isCounterIncrement() &&
           " Bottom phony edge"
           " did not contain a path counter increment.");

    // split edge has yet to be initialized
    if (!instrumentNode->getEndingPathNumber()) {
      instrumentNode->setStartingPathNumber(createIncrementConstant(0, 32));
      instrumentNode->setEndingPathNumber(createIncrementConstant(0, 32));
    }

    auto insertPoint = atBeginning
                           ? &*instrumentNode->getBlock()->getFirstInsertionPt()
                           : instrumentNode->getBlock()->getTerminator();

    // add information from the bottom edge, if it exists
    if (bottom->getIncrement()) {
      Value *newpn = BinaryOperator::Create(
          Instruction::Add, instrumentNode->getStartingPathNumber(),
          createIncrementConstant(bottom), "pathNumber", insertPoint);
      instrumentNode->setEndingPathNumber(newpn);
    }

    insertCounterIncrement(instrumentNode->getEndingPathNumber(), insertPoint,
                           dag, FTEntry);

    if (atBeginning)
      instrumentNode->setStartingPathNumber(createIncrementConstant(top));

    instrumentNode->setEndingPathNumber(createIncrementConstant(top));

    // Check for path counter increments
    if (top->isCounterIncrement()) {
      insertCounterIncrement(instrumentNode->getEndingPathNumber(),
                             instrumentNode->getBlock()->getTerminator(), dag,
                             FTEntry);
      instrumentNode->setEndingPathNumber(0);
    }
  }

  // Insert instrumentation if this is a normal edge
  else {
    auto insertPoint = atBeginning
                           ? &*instrumentNode->getBlock()->getFirstInsertionPt()
                           : instrumentNode->getBlock()->getTerminator();

    if (edge->isInitialization()) { // initialize path number
      instrumentNode->setEndingPathNumber(createIncrementConstant(edge));
    } else if (edge->getIncrement()) { // increment path number
      Value *newpn = BinaryOperator::Create(
          Instruction::Add, instrumentNode->getStartingPathNumber(),
          createIncrementConstant(edge), "pathNumber", insertPoint);
      instrumentNode->setEndingPathNumber(newpn);

      if (atBeginning)
        instrumentNode->setStartingPathNumber(newpn);
    }

    // Check for path counter increments
    if (edge->isCounterIncrement()) {
      insertCounterIncrement(instrumentNode->getEndingPathNumber(), insertPoint,
                             dag, FTEntry);
      instrumentNode->setEndingPathNumber(0);
    }
  }

  // Push it along
  if (nextSourceNode && instrumentNode->getEndingPathNumber())
    pushValueIntoNode(instrumentNode, nextSourceNode);

  // Add all the successors
  for (BLEdgeIterator next = targetNode->succBegin(),
                      end = targetNode->succEnd();
       next != end; next++) {
    // So long as it is un-instrumented, add it to the list
    if (!((BLInstrumentationEdge *)(*next))->hasInstrumentation()) {
      insertInstrumentationStartingAt((BLInstrumentationEdge *)*next, dag,
                                      FTEntry);
    } else {
      LLVM_DEBUG(dbgs() << "  Edge " << *(BLInstrumentationEdge *)(*next)
                        << " already instrumented.\n");
    }
  }
}

// Inserts instrumentation according to the marked edges in dag.  Phony edges
// must be unlinked from the DAG, but accessible from the backedges.  Dag
// must have initializations, path number increments, and counter increments
// present.
//
// Counter storage is created here.
void PathProfiler::insertInstrumentation(BLInstrumentationDag &dag,
                                         Constant *FTEntry, Module &M) {
  BLInstrumentationEdge *exitRootEdge =
      (BLInstrumentationEdge *)dag.getExitRootEdge();
  insertInstrumentationStartingAt(exitRootEdge, &dag, FTEntry);

  // Iterate through each call edge and apply the appropriate hash increment
  // and decrement functions
  BLEdgeVector callEdges = dag.getCallPhonyEdges();
  for (BLEdgeIterator edge = callEdges.begin(), end = callEdges.end();
       edge != end; edge++) {
    BLInstrumentationNode *node = (BLInstrumentationNode *)(*edge)->getSource();
    BasicBlock::iterator insertPoint = node->getBlock()->getFirstInsertionPt();

    // Find the first function call
    while (((Instruction &)(*insertPoint)).getOpcode() != Instruction::Call)
      insertPoint++;

    LLVM_DEBUG(dbgs() << "\nInstrumenting method call block '"
                      << node->getBlock()->getName() << "'\n");
    LLVM_DEBUG(dbgs() << "   Path number initialized: "
                      << ((node->getStartingPathNumber()) ? "yes" : "no")
                      << "\n");

    Value *newpn;
    if (node->getStartingPathNumber()) {
      long inc = ((BLInstrumentationEdge *)(*edge))->getIncrement();
      if (inc)
        newpn = BinaryOperator::Create(
            Instruction::Add, node->getStartingPathNumber(),
            createIncrementConstant(inc, 32), "pathNumber", &*insertPoint);
      else
        newpn = node->getStartingPathNumber();
    } else {
      newpn = (Value *)createIncrementConstant(
          ((BLInstrumentationEdge *)(*edge))->getIncrement(), 32);
    }

    insertCounterIncrement(newpn, &*insertPoint, &dag, FTEntry);
    insertCounterIncrement(newpn, node->getBlock()->getTerminator(), &dag,
                           FTEntry, false);
  }
}

// Entry point of the module
Constant *PathProfiler::runOnFunction(Function &F, Module &M) {
  // Build DAG from CFG
  BLInstrumentationDag dag = BLInstrumentationDag(F);
  dag.init();

  // give each path a unique integer value
  dag.calculatePathNumbers();

  // modify path increments to increase the efficiency
  // of instrumentation
  dag.calculateSpanningTree();
  dag.calculateChordIncrements();
  dag.pushInitialization();
  dag.pushCounters();
  dag.unlinkPhony();

  // potentially generate .dot graph for the dag
  if (DotPathDag) {
    dag.generateDotGraph();
  }

  if (dag.getNumberOfPaths() > HashThreshold) {
    WARNF("Too many paths in %s to track in an array. Skipping",
          F.getName().str().c_str());
    return nullptr;
  }

  // Create the entry for the function table
  auto *FunctionEntry = ConstantStruct::get(
      FTEntryTy, createIncrementConstant(dag.getNumberOfPaths(), 32),
      Constant::getNullValue(VoidPtrTy));
  auto *FTEntryGV = new GlobalVariable(
      M, FTEntryTy, false, GlobalVariable::PrivateLinkage, FunctionEntry,
      "__path_profiling_entry." + F.getName());
  auto *Counter = ConstantExpr::getInBoundsGetElementPtr(
      FTEntryTy, FTEntryGV,
      ArrayRef<Constant *>(
          {Constant::getNullValue(Int32Ty), ConstantInt::get(Int32Ty, 1)}));
  dag.setCounterArray(Counter);

  if (TargetTriple.supportsCOMDAT() && !F.isInterposable()) {
    if (auto Comdat =
            GetOrCreateFunctionComdat(F, TargetTriple, CurModuleUniqueId)) {
      FTEntryGV->setComdat(Comdat);
    }
  }

  FTEntryGV->setSection(getSectionName(kPathProfSectionName));
  FTEntryGV->setAlignment(Align(DL->getTypeStoreSize(FTEntryTy)));
  GlobalsToAppendToUsed.push_back(FTEntryGV);
  GlobalsToAppendToCompilerUsed.push_back(FTEntryGV);
  auto *MD = MDNode::get(F.getContext(), ValueAsMetadata::get(&F));
  FTEntryGV->addMetadata(LLVMContext::MD_associated, *MD);

  // Do instrumentation
  insertInstrumentation(dag, FTEntryGV, M);
  numInstrumentedFunctions++;

  return FTEntryGV;
}

bool PathProfiler::runOnModule(Module &M) {
  Context = &M.getContext();
  DL = &M.getDataLayout();
  CurModuleUniqueId = getUniqueModuleId(&M);
  TargetTriple = Triple(M.getTargetTriple());

  Int8Ty = Type::getInt8Ty(*Context);
  Int32Ty = Type::getInt32Ty(*Context);
  VoidPtrTy = Type::getInt8PtrTy(*Context);
  FTEntryTy = StructType::create(*Context, {Int32Ty, VoidPtrTy}, "FTEntry");

  numInstrumentedFunctions = 0;

  debugFunction =
      M.getOrInsertFunction("__path_profiler_dbg", Type::getVoidTy(*Context),
                            PointerType::getUnqual(FTEntryTy), Int32Ty);

  unsigned functionNumber = 0;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    LLVM_DEBUG(dbgs() << "Function: " << F.getName() << "\n");
    functionNumber++;

    // Set function number
    currentFunctionNumber = functionNumber;

    // Analyze and instrument function
    runOnFunction(F, M);
  }

  createInitCallForSection(M);
  if (TargetTriple.isOSBinFormatMachO()) {
    appendToUsed(M, GlobalsToAppendToUsed);
  }
  appendToCompilerUsed(M, GlobalsToAppendToCompilerUsed);

  return true;
}

std::string PathProfiler::getSectionStart(const std::string &Section) const {
  if (TargetTriple.isOSBinFormatMachO()) {
    return "\1section$start$__DATA$__" + Section;
  }
  return "__start___" + Section;
}

std::string PathProfiler::getSectionEnd(const std::string &Section) const {
  if (TargetTriple.isOSBinFormatMachO()) {
    return "\1section$end$__DATA$__" + Section;
  }
  return "__stop___" + Section;
}

std::string PathProfiler::getSectionName(const std::string &Section) const {
  if (TargetTriple.isOSBinFormatCOFF()) {
    return ".PATHPROF$CM";
  }
  if (TargetTriple.isOSBinFormatMachO()) {
    return "__DATA,__" + Section;
  }
  return "__" + Section;
}

std::pair<Value *, Value *>
PathProfiler::CreateSecStartEnd(Module &M, const char *Section, Type *Ty) {
  GlobalVariable *SecStart = new GlobalVariable(
      M, Ty->getPointerElementType(), false, GlobalVariable::ExternalLinkage,
      nullptr, getSectionStart(Section));
  SecStart->setVisibility(GlobalValue::HiddenVisibility);
  GlobalVariable *SecEnd = new GlobalVariable(
      M, Ty->getPointerElementType(), false, GlobalVariable::ExternalLinkage,
      nullptr, getSectionEnd(Section));
  SecEnd->setVisibility(GlobalValue::HiddenVisibility);

  if (!TargetTriple.isOSBinFormatCOFF()) {
    return {SecStart, SecEnd};
  }

  auto *Int8Ty = Type::getInt8Ty(*Context);
  auto *IntPtrTy = Type::getIntNTy(*Context, DL->getPointerSizeInBits());

  IRBuilder<> IRB(*Context);
  auto SecStartI8Ptr = IRB.CreatePointerCast(SecStart, VoidPtrTy);
  auto GEP = IRB.CreateGEP(Int8Ty, SecStartI8Ptr,
                           ConstantInt::get(IntPtrTy, sizeof(uint64_t)));
  return {IRB.CreatePointerCast(GEP, Ty), SecEnd};
}

Function *PathProfiler::createInitCallForSection(Module &M) {
  auto *Ty = PointerType::getUnqual(FTEntryTy);
  const auto &[SecStart, SecEnd] =
      CreateSecStartEnd(M, kPathProfSectionName, Ty);

  Function *CtorFunc;
  std::tie(CtorFunc, std::ignore) = createSanitizerCtorAndInitFunctions(
      M, kPathProfCtorName, kPathProfInitFunctionName, {Ty, Ty},
      {SecStart, SecEnd});
  assert(CtorFunc->getName() == kPathProfCtorName);

  if (TargetTriple.supportsCOMDAT()) {
    // Use comdat to dedup CtorFunc.
    CtorFunc->setComdat(M.getOrInsertComdat(kPathProfCtorName));
    appendToGlobalCtors(M, CtorFunc, kPathProfCtorPriority, CtorFunc);
  } else {
    appendToGlobalCtors(M, CtorFunc, kPathProfCtorPriority);
  }

  if (TargetTriple.isOSBinFormatCOFF()) {
    CtorFunc->setLinkage(GlobalValue::WeakODRLinkage);
    appendToUsed(M, CtorFunc);
  }

  return CtorFunc;
}

// If this edge is a critical edge, then inserts a node at this edge.
// This edge becomes the first edge, and a new BallLarusEdge is created.
// Returns true if the edge was split
bool PathProfiler::splitCritical(BLInstrumentationEdge *edge,
                                 BLInstrumentationDag *dag) {
  unsigned succNum = edge->getSuccessorNumber();
  BallLarusNode *sourceNode = edge->getSource();
  BallLarusNode *targetNode = edge->getTarget();
  BasicBlock *sourceBlock = sourceNode->getBlock();
  BasicBlock *targetBlock = targetNode->getBlock();

  if (sourceBlock == nullptr || targetBlock == nullptr ||
      sourceNode->getNumberSuccEdges() <= 1 ||
      targetNode->getNumberPredEdges() == 1) {
    return false;
  }

  auto *terminator = sourceBlock->getTerminator();

  if (SplitCriticalEdge(terminator, succNum)) {
    BasicBlock *newBlock = terminator->getSuccessor(succNum);
    dag->splitUpdate(edge, newBlock);
    return true;
  } else
    return false;
}

//
// Legacy pass manager registration
//

static RegisterPass<PathProfiler> X("path-profiler",
                                    "Ball-Larus path profiling", false, false);

static void registerPathProfiler(const PassManagerBuilder &,
                                 legacy::PassManagerBase &PM) {
  PM.add(new PathProfiler());
}

static RegisterStandardPasses
    RegisterPathProfiler(PassManagerBuilder::EP_ModuleOptimizerEarly,
                         registerPathProfiler);
static RegisterStandardPasses
    registerPathProfiler0(PassManagerBuilder::EP_EnabledOnOptLevel0,
                          registerPathProfiler);
