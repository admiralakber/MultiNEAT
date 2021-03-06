#ifndef _GENOME_H
#define _GENOME_H

///////////////////////////////////////////////////////////////////////////////////////////
//    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published
//    by the Free Software Foundation, either version 3 of the License, or (at
//    your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
//
//    Contact info:
//
//    Peter Chervenski < spookey@abv.bg >
//    Shane Ryan < shane.mcdonald.ryan@gmail.com >
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Genome.h
// Description: Definition for the Genome class.
///////////////////////////////////////////////////////////////////////////////

#include <boost/shared_ptr.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/topological_sort.hpp>

#include <queue>
#include <vector>

#include <MultiNEAT/Assert.hh>
#include <MultiNEAT/Genes.hh>
#include <MultiNEAT/Innovation.hh>
#include <MultiNEAT/NeuralNetwork.hh>
#include <MultiNEAT/PhenotypeBehavior.hh>
#include <MultiNEAT/Random.hh>
#include <MultiNEAT/Substrate.hh>

namespace NEAT {

//////////////////////////////////////////////
// The Genome class
//////////////////////////////////////////////

// forward
class Innovation;

class InnovationDatabase;

class PhenotypeBehavior;

extern ActivationFunction GetRandomActivation(Parameters &a_Parameters,
                                              RNG &a_RNG);

namespace bs = boost;

typedef bs::adjacency_list<bs::vecS, bs::vecS, bs::directedS> Graph;
typedef bs::graph_traits<Graph>::vertex_descriptor Vertex;

class Genome {
  /////////////////////
  // Members
  /////////////////////
private:
  // ID of genome
  unsigned int m_ID;

  // How many inputs/outputs
  unsigned int m_NumInputs;
  unsigned int m_NumOutputs;

  // The genome's fitness score
  double m_Fitness;

  // The genome's adjusted fitness score
  double m_AdjustedFitness;

  // The depth of the network
  unsigned int m_Depth;

  // how many individuals this genome should spawn
  double m_OffspringAmount;

  ////////////////////
  // Private methods

  // Returns true if the specified neuron ID is present in the genome
  bool HasNeuronID(int a_id) const;

  // Returns true if the specified link is present in the genome
  bool HasLink(int a_n1id, int a_n2id) const;

  // Returns true if the specified link is present in the genome
  bool HasLinkByInnovID(int a_id) const;

  // Removes the link with the specified innovation ID
  void RemoveLinkGene(int a_innovid);

  // Remove node
  // Links connected to this node are also removed
  void RemoveNeuronGene(int a_id);

  // Returns the count of links inputting from the specified neuron ID
  int LinksInputtingFrom(int a_id) const;

  // Returns the count of links outputting to the specified neuron ID
  int LinksOutputtingTo(int a_id) const;

  // A recursive function returning the max depth from the specified neuron to
  // the inputs
  unsigned int NeuronDepth(int a_NeuronID, unsigned int a_Depth);

  // Returns true is the specified neuron ID is a dead end or isolated
  bool IsDeadEndNeuron(int a_id) const;

public:
  // The two lists of genes
  std::vector<NeuronGene> m_NeuronGenes;
  std::vector<LinkGene> m_LinkGenes;

  // To have traits that belong to the genome itself
  Gene m_GenomeGene;

  // tells whether this genome was evaluated already
  // used in steady state evolution
  bool m_Evaluated;

  // the initial genome complexity
  int m_initial_num_neurons;
  int m_initial_num_links;

  // A pointer to a class representing the phenotype's behavior
  // Used in novelty searches
  PhenotypeBehavior *m_PhenotypeBehavior;
  // A Python object behavior

  ////////////////////////////
  // Constructors
  ////////////////////////////

  Genome();

  // copy constructor
  Genome(const Genome &a_g);

  // assignment operator
  Genome &operator=(const Genome &a_g);

  // comparison operator (nessesary for boost::python)
  // todo: implement a better comparison technique
  bool operator==(Genome const &other) const { return m_ID == other.m_ID; }

  // Builds this genome from a file
  Genome(const char *a_filename);

  // Builds this genome from an opened file
  Genome(std::ifstream &a_DataFile);

  // This creates a CTRNN fully-connected genome
  Genome(unsigned int a_ID, unsigned int a_NumInputs, unsigned int a_NumHidden,
         unsigned int a_NumOutputs, ActivationFunction a_OutputActType,
         ActivationFunction a_HiddenActType, const Parameters &a_Parameters);

  // This creates a standart minimal genome - perceptron-like structure
  Genome(unsigned int a_ID, unsigned int a_NumInputs,
         unsigned int a_NumHidden, // ignored for seed_type == 0, specifies
                                   // number of hidden units if seed_type == 1
         unsigned int a_NumOutputs, bool a_FS_NEAT,
         ActivationFunction a_OutputActType, ActivationFunction a_HiddenActType,
         unsigned int a_SeedType, const Parameters &a_Parameters,
         unsigned int a_NumLayers);

  /////////////
  // Other possible constructors for different types of networks go here
  // TODO

  ////////////////////////////
  // Destructor
  ////////////////////////////

  ////////////////////////////
  // Methods
  ////////////////////////////

  ////////////////////
  // Accessor methods

  NeuronGene GetNeuronByID(int a_ID) const;

  NeuronGene GetNeuronByIndex(int a_idx) const;

  LinkGene GetLinkByInnovID(int a_ID) const;

  LinkGene GetLinkByIndex(int a_idx) const;

  // A little helper function to find the index of a neuron, given its ID
  int GetNeuronIndex(int a_id) const;

  // A little helper function to find the index of a link, given its innovation
  // ID
  int GetLinkIndex(int a_innovid) const;

  unsigned int NumNeurons() const {
    return static_cast<unsigned int>(m_NeuronGenes.size());
  }

  unsigned int NumLinks() const {
    return static_cast<unsigned int>(m_LinkGenes.size());
  }

  unsigned int NumInputs() const { return m_NumInputs; }

  unsigned int NumOutputs() const { return m_NumOutputs; }

  void SetNeuronXY(unsigned int a_idx, int a_x, int a_y);

  void SetNeuronX(unsigned int a_idx, int a_x);

  void SetNeuronY(unsigned int a_idx, int a_y);

  double GetFitness() const;

  double GetAdjFitness() const;

  void SetFitness(double a_f);

  void SetAdjFitness(double a_af);

  unsigned int GetID() const;

  void SetID(unsigned int a_id);

  unsigned int GetDepth() const;

  void SetDepth(unsigned int a_d);

  // Returns true if there is any dead end in the network
  bool HasDeadEnds() const;

  // Returns true if there is any looping path in the network
  bool HasLoops();

  bool FailsConstraints(const Parameters &a_Parameters) {
    bool fails = false;

    if (HasDeadEnds() || (NumLinks() == 0)) {
      return true; // no reason to continue
    }

    if ((HasLoops() && (a_Parameters.AllowLoops == false))) {
      return true;
    }

    // Custom constraints
    if (a_Parameters.CustomConstraints != NULL) {
      if (a_Parameters.CustomConstraints(*this)) {
        return true;
      }
    }

    // for Python-based custom constraint callbacks
    // add more constraints here
    return false;
  }

  double GetOffspringAmount() const;

  void SetOffspringAmount(double a_oa);

  // This builds a fastnetwork structure out from the genome
  void BuildPhenotype(NeuralNetwork &net);

  // Projects the phenotype's weights back to the genome
  void DerivePhenotypicChanges(NeuralNetwork &a_Net);

  ////////////
  // Other possible methods for building a phenotype go here
  // Like CPPN/HyperNEAT stuff
  ////////////
  void BuildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst);

  // Saves this genome to a file
  void Save(const char *a_filename);

  // Saves this genome to an already opened file for writing
  void Save(FILE *a_fstream);

  void PrintTraits(std::map<std::string, Trait> &traits);
  void PrintAllTraits();

  // returns the max neuron ID
  int GetLastNeuronID() const;

  // returns the max innovation Id
  int GetLastInnovationID() const;

  // Sorts the genes of the genome
  // The neurons by IDs and the links by innovation numbers.
  void SortGenes();

  // overload '<' used for sorting. From fittest to poorest.
  friend bool operator<(const Genome &a_lhs, const Genome &a_rhs) {
    return (a_lhs.m_Fitness > a_rhs.m_Fitness);
  }

  // Returns true if this genome and a_G are compatible (belong in the same
  // species)
  bool IsCompatibleWith(Genome &a_G, Parameters &a_Parameters);

  // returns the absolute compatibility distance between this genome and a_G
  double CompatibilityDistance(Genome &a_G, Parameters &a_Parameters);

  // Calculates the network depth
  void CalculateDepth();

  ////////////
  // Mutation
  ////////////

  // Adds a new neuron to the genome
  // returns true if succesful
  bool Mutate_AddNeuron(InnovationDatabase &a_Innovs,
                        const Parameters &a_Parameters, RNG &a_RNG);

  // Adds a new link to the genome
  // returns true if succesful
  bool Mutate_AddLink(InnovationDatabase &a_Innovs,
                      const Parameters &a_Parameters, RNG &a_RNG);

  // Remove a random link from the genome
  // A cleanup procedure is invoked so any dead-ends or stranded neurons are
  // also deleted returns true if succesful
  bool Mutate_RemoveLink(RNG &a_RNG);

  // Removes a hidden neuron having only one input and only one output with
  // a direct link between them.
  bool Mutate_RemoveSimpleNeuron(InnovationDatabase &a_Innovs, RNG &a_RNG);

  // Perturbs the weights
  bool Mutate_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG);

  // Set all link weights to random values between [-R .. R]
  void Randomize_LinkWeights(double a_Range, RNG &a_RNG);

  // Set all traits to random values
  void Randomize_Traits(const Parameters &a_Parameters, RNG &a_RNG);

  // Perturbs the A parameters of the neuron activation functions
  bool Mutate_NeuronActivations_A(const Parameters &a_Parameters, RNG &a_RNG);

  // Perturbs the B parameters of the neuron activation functions
  bool Mutate_NeuronActivations_B(const Parameters &a_Parameters, RNG &a_RNG);

  // Changes the activation function type for a random neuron
  bool Mutate_NeuronActivation_Type(const Parameters &a_Parameters, RNG &a_RNG);

  // Perturbs the neuron time constants
  bool Mutate_NeuronTimeConstants(const Parameters &a_Parameters, RNG &a_RNG);

  // Perturbs the neuron biases
  bool Mutate_NeuronBiases(const Parameters &a_Parameters, RNG &a_RNG);

  // Perturbs the neuron traits
  bool Mutate_NeuronTraits(const Parameters &a_Parameters, RNG &a_RNG);

  // Perturbs the link traits
  bool Mutate_LinkTraits(const Parameters &a_Parameters, RNG &a_RNG);

  // Perturbs the genome traits
  bool Mutate_GenomeTraits(const Parameters &a_Parameters, RNG &a_RNG);

  ///////////
  // Mating
  ///////////

  // Mate this genome with dad and return the baby
  // If this is multipoint mating, genes are inherited randomly
  // If the a_averagemating bool is true, then the genes are averaged
  // Disjoint and excess genes are inherited from the fittest parent
  // If fitness is equal, the smaller genome is assumed to be the better one
  Genome Mate(Genome &a_dad, bool a_averagemating, bool a_interspecies,
              RNG &a_RNG, Parameters &a_Parameters);

  //////////
  // Utility
  //////////

  // Search the genome for isolated structure and clean it up
  // Returns true is something was removed
  bool Cleanup();

  ////////////////////
  // new stuff
  bool IsEvaluated() const;

  void SetEvaluated();

  void ResetEvaluated();

  /////////////////////////////////////////////
  // Evolvable Substrate HyperNEAT
  ////////////////////////////////////////////

  // A connection between two points. Stores weight and the coordinates of the
  // points
  struct TempConnection {
    std::vector<double> source;
    std::vector<double> target;
    double weight;

    TempConnection() {
      source.reserve(3);
      target.reserve(3);
      weight = 0;
    }

    TempConnection(std::vector<double> t_source, std::vector<double> t_target,
                   double t_weight) {
      source = t_source;
      target = t_target;
      weight = t_weight;
      source.reserve(3);
      target.reserve(3);
    }

    ~TempConnection(){};

    bool operator==(const TempConnection &rhs) const {
      return (source == rhs.source && target == rhs.target);
    }

    bool operator!=(const TempConnection &rhs) const {
      return (source != rhs.source && target != rhs.target);
    }
  };

  // A quadpoint in the HyperCube.
  struct QuadPoint {
    double x;
    double y;
    double z;
    double width;
    double weight;
    double height;
    double variance;
    int level;
    // Do I use this?
    double leo;

    std::vector<boost::shared_ptr<QuadPoint>> children;

    QuadPoint() {
      x = y = z = width = height = weight = variance = leo = 0;
      level = 0;
      children.reserve(4);
    }

    QuadPoint(double t_x, double t_y, double t_width, double t_height,
              int t_level) {
      x = t_x;
      y = t_y;
      z = 0.0;
      width = t_width;
      height = t_height;
      level = t_level;
      weight = 0.0;
      leo = 0.0;
      variance = 0.0;
      children.reserve(4);
      children.clear();
    }

    // Mind the Z
    QuadPoint(double t_x, double t_y, double t_z, double t_width,
              double t_height, int t_level) {
      x = t_x;
      y = t_y;
      z = t_z;
      width = t_width;
      height = t_height;
      level = t_level;
      weight = 0.0;
      variance = 0.0;
      leo = 0.0;
      children.reserve(4);
      children.clear();
    }

    ~QuadPoint(){};
  };

  void BuildESHyperNEATPhenotype(NeuralNetwork &a_net, Substrate &subst,
                                 Parameters &params);

  void DivideInitialize(const std::vector<double> &node,
                        boost::shared_ptr<QuadPoint> &root, NeuralNetwork &cppn,
                        Parameters &params, const bool &outgoing,
                        const double &z_coord);

  void PruneExpress(const std::vector<double> &node,
                    boost::shared_ptr<QuadPoint> &root, NeuralNetwork &cppn,
                    Parameters &params,
                    std::vector<Genome::TempConnection> &connections,
                    const bool &outgoing);

  void CollectValues(std::vector<double> &vals,
                     boost::shared_ptr<QuadPoint> &point);

  double Variance(boost::shared_ptr<QuadPoint> &point);

  void Clean_Net(std::vector<Connection> &connections, unsigned int input_count,
                 unsigned int output_count, unsigned int hidden_count);
};

#define DBG(x)                                                                 \
  { std::cerr << x << std::endl; }

} // namespace NEAT

#endif
