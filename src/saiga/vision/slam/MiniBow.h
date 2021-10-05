/**
 * Original File: TemplatedVocabulary.h
 * Original Author: Dorian Galvez-Lopez
 *
 * Modified by: Darius Rückert
 * Modifications:
 *  - Moved everything into this single header file
 *  - Removed support for non-ORB feature descriptors
 *  - Optimized loading, saving, matching
 *  - Removed dependency to opencv
 *
 * Original License: BSD-like
 *          https://github.com/dorian3d/DBoW2/blob/master/LICENSE.txt
 * License of modifications: MIT
 *          https://github.com/darglein/DBoW2/blob/master/LICENSE.txt
 *
 */
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace MiniBow
{
/// Id of words
typedef unsigned int WordId;

/// Value of a word
typedef double WordValue;

/// Id of nodes in the vocabulary treee
typedef unsigned int NodeId;

/// L-norms for normalization
enum LNorm
{
    L1,
    L2
};

/// Weighting type
enum WeightingType
{
    TF_IDF,
    TF,
    IDF,
    BINARY
};



class FORB
{
   public:
    using TDescriptor = std::array<uint64_t, 4>;
    typedef const TDescriptor* pDescriptor;
    static const int L = 32;
    /**
     * Calculates the mean value of a set of descriptors
     * @param descriptors
     * @param mean mean descriptor
     */
    static void meanValue(const std::vector<pDescriptor>& descriptors, TDescriptor& mean)
    {
        if (descriptors.empty())
        {
            return;
        }
        else if (descriptors.size() == 1)
        {
            mean = *descriptors[0];
        }
        else
        {
            std::vector<int> sum(FORB::L * 8, 0);

            for (size_t i = 0; i < descriptors.size(); ++i)
            {
                const auto& d          = *descriptors[i];
                const unsigned char* p = (const unsigned char*)d.data();
                for (int j = 0; j < 32; ++j, ++p)
                {
                    if (*p & (1 << 7)) ++sum[j * 8];
                    if (*p & (1 << 6)) ++sum[j * 8 + 1];
                    if (*p & (1 << 5)) ++sum[j * 8 + 2];
                    if (*p & (1 << 4)) ++sum[j * 8 + 3];
                    if (*p & (1 << 3)) ++sum[j * 8 + 4];
                    if (*p & (1 << 2)) ++sum[j * 8 + 5];
                    if (*p & (1 << 1)) ++sum[j * 8 + 6];
                    if (*p & (1)) ++sum[j * 8 + 7];
                }
            }
            std::fill(mean.begin(), mean.end(), 0);
            unsigned char* p = (unsigned char*)mean.data();
            const int N2     = (int)descriptors.size() / 2 + descriptors.size() % 2;
            for (size_t i = 0; i < sum.size(); ++i)
            {
                if (sum[i] >= N2)
                {
                    // set bit
                    *p |= 1 << (7 - (i % 8));
                }
                if (i % 8 == 7) ++p;
            }
        }
    }

    /**
     * Calculates the distance between two descriptors
     * @param a
     * @param b
     * @return distance
     */
#if !defined(WIN32) && EIGEN_ARCH_i386_OR_x86_64
    static inline int popcnt64(uint64_t x)
    {
        __asm__("popcnt %1, %0" : "=r"(x) : "0"(x));
        return x;
    }
#else
    static inline int popcnt64(uint64_t v)
    {
        v = v - ((v >> 1) & (uint64_t) ~(uint64_t)0 / 3);
        v = (v & (uint64_t) ~(uint64_t)0 / 15 * 3) + ((v >> 2) & (uint64_t) ~(uint64_t)0 / 15 * 3);
        v = (v + (v >> 4)) & (uint64_t) ~(uint64_t)0 / 255 * 15;
        return (uint64_t)(v * ((uint64_t) ~(uint64_t)0 / 255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT;
    }
#endif

    static double distance(const TDescriptor& a, const TDescriptor& b)
    {
        auto pa  = (uint64_t*)a.data();
        auto pb  = (uint64_t*)b.data();
        int dist = 0;
        for (int i = 0; i < 4; i++, pa++, pb++)
        {
            uint64_t v = *pa ^ *pb;
            dist += popcnt64(v);
        }
        return dist;
    }
};
/// Vector of words to represent images
class BowVector : public std::map<WordId, WordValue>
{
   public:
    /**
     * Adds a value to a word value existing in the vector, or creates a new
     * word with the given value
     * @param id word id to look for
     * @param v value to create the word with, or to add to existing word
     */
    void addWeight(WordId id, WordValue v)
    {
        BowVector::iterator vit = this->lower_bound(id);

        if (vit != this->end() && !(this->key_comp()(id, vit->first)))
        {
            vit->second += v;
        }
        else
        {
            this->insert(vit, BowVector::value_type(id, v));
        }
    }

    /**
     * Adds a word with a value to the vector only if this does not exist yet
     * @param id word id to look for
     * @param v value to give to the word if this does not exist
     */
    void addIfNotExist(WordId id, WordValue v)
    {
        BowVector::iterator vit = this->lower_bound(id);

        if (vit == this->end() || (this->key_comp()(id, vit->first)))
        {
            this->insert(vit, BowVector::value_type(id, v));
        }
    }

    /**
     * L1-Normalizes the values in the vector
     * @param norm_type norm used
     */
    void normalize()
    {
        double norm = 0.0;
        BowVector::iterator it;
        {
            for (it = begin(); it != end(); ++it) norm += std::abs(it->second);
        }
        if (norm > 0.0)
        {
            for (it = begin(); it != end(); ++it) it->second /= norm;
        }
    }
};
class FeatureVector : public std::map<NodeId, std::vector<int>>
{
   public:
    void addFeature(NodeId id, int i_feature)
    {
        FeatureVector::iterator vit = this->lower_bound(id);

        if (vit != this->end() && vit->first == id)
        {
            vit->second.push_back(i_feature);
        }
        else
        {
            vit = this->insert(vit, FeatureVector::value_type(id, std::vector<int>()));
            vit->second.push_back(i_feature);
        }
    }
};

class L1Scoring
{
   public:
    //    static constexpr inline int id      = 0;
    enum
    {
        id = 0
    };
    static constexpr bool mustNormalize = true;
    static double score(const BowVector& v1, const BowVector& v2)
    {
        BowVector::const_iterator v1_it, v2_it;
        const BowVector::const_iterator v1_end = v1.end();
        const BowVector::const_iterator v2_end = v2.end();
        v1_it                                  = v1.begin();
        v2_it                                  = v2.begin();
        double score                           = 0;
        while (v1_it != v1_end && v2_it != v2_end)
        {
            const WordValue& vi = v1_it->second;
            const WordValue& wi = v2_it->second;

            if (v1_it->first == v2_it->first)
            {
                score += std::abs(vi - wi) - std::abs(vi) - std::abs(wi);
                ++v1_it;
                ++v2_it;
            }
            else if (v1_it->first < v2_it->first)
            {
                v1_it = v1.lower_bound(v2_it->first);
            }
            else
            {
                v2_it = v2.lower_bound(v1_it->first);
            }
        }
        score = -score / 2.0;

        return score;  // [0..1]
    }
};

/// @param TDescriptor class of descriptor
/// @param F class of descriptor functions
template <class TDescriptor, class F, class Scoring>
/// Generic Vocabulary
class TemplatedVocabulary
{
   public:
    /**
     * Initiates an empty vocabulary
     * @param k branching factor
     * @param L depth levels
     * @param weighting weighting type
     * @param scoring scoring type
     */
    TemplatedVocabulary(int k = 10, int L = 5, WeightingType weighting = TF_IDF);

    /**
     * Creates the vocabulary by loading a file
     * @param filename
     */
    TemplatedVocabulary(const std::string& filename);


    /**
     * Destructor
     */
    virtual ~TemplatedVocabulary();


    /**
     * Creates a vocabulary from the training features with the already
     * defined parameters
     * @param training_features
     */
    virtual void create(const std::vector<std::vector<TDescriptor>>& training_features);

    /**
     * Creates a vocabulary from the training features, setting the branching
     * factor and the depth levels of the tree
     * @param training_features
     * @param k branching factor
     * @param L depth levels
     */
    virtual void create(const std::vector<std::vector<TDescriptor>>& training_features, int k, int L);

    /**
     * Creates a vocabulary from the training features, setting the branching
     * factor nad the depth levels of the tree, and the weighting and scoring
     * schemes
     */
    virtual void create(const std::vector<std::vector<TDescriptor>>& training_features, int k, int L,
                        WeightingType weighting);

    /**
     * Returns the number of words in the vocabulary
     * @return number of words
     */
    virtual inline unsigned int size() const;

    /**
     * Returns whether the vocabulary is empty (i.e. it has not been trained)
     * @return true iff the vocabulary is empty
     */
    virtual inline bool empty() const;

    /**
     * Transforms a set of descriptores into a bow vector
     * @param features
     * @param v (out) bow vector of weighted words
     */
    virtual void transform(const std::vector<TDescriptor>& features, BowVector& v) const;

    /**
     * Transform a set of descriptors into a bow vector and a feature vector
     * @param features
     * @param v (out) bow vector
     * @param fv (out) feature vector of nodes and feature indexes
     * @param levelsup levels to go up the vocabulary tree to get the node index
     */
    virtual void transform(const std::vector<TDescriptor>& features, BowVector& v, FeatureVector& fv,
                           int levelsup) const;
    virtual void transformOMP(const std::vector<TDescriptor>& features, BowVector& v, FeatureVector& fv, int levelsup);

    // shared OMP variables
    using TransformResult = std::tuple<WordId, NodeId, WordValue>;
    int N = 0;
    std::vector<TransformResult> transformedFeatures;

    /**
     * Transforms a single feature into a word (without weight)
     * @param feature
     * @return word id
     */
    virtual WordId transform(const TDescriptor& feature) const;

    /**
     * Returns the score of two vectors
     * @param a vector
     * @param b vector
     * @return score between vectors
     * @note the vectors must be already sorted and normalized if necessary
     */
    inline double score(const BowVector& a, const BowVector& b) const;

    /**
     * Returns the id of the node that is "levelsup" levels from the word given
     * @param wid word id
     * @param levelsup 0..L
     * @return node id. if levelsup is 0, returns the node id associated to the
     *   word id
     */
    virtual NodeId getParentNode(WordId wid, int levelsup) const;

    /**
     * Returns the ids of all the words that are under the given node id,
     * by traversing any of the branches that goes down from the node
     * @param nid starting node id
     * @param words ids of words
     */
    void getWordsFromNode(NodeId nid, std::vector<WordId>& words) const;

    /**
     * Returns the branching factor of the tree (k)
     * @return k
     */
    inline int getBranchingFactor() const { return m_k; }

    /**
     * Returns the depth levels of the tree (L)
     * @return L
     */
    inline int getDepthLevels() const { return m_L; }

    /**
     * Returns the real depth levels of the tree on average
     * @return average of depth levels of leaves
     */
    float getEffectiveLevels() const;

    /**
     * Returns the descriptor of a word
     * @param wid word id
     * @return descriptor
     */
    virtual inline TDescriptor getWord(WordId wid) const;

    /**
     * Returns the weight of a word
     * @param wid word id
     * @return weight
     */
    virtual inline WordValue getWordWeight(WordId wid) const;

    /**
     * Returns the weighting method
     * @return weighting method
     */
    inline WeightingType getWeightingType() const { return m_weighting; }


    /**
     * Changes the weighting method
     * @param type new weighting type
     */
    inline void setWeightingType(WeightingType type);

    /**
     * Changes the scoring method
     * @param type new scoring type
     */



    virtual void saveRaw(const std::string& file) const;
    virtual void loadRaw(const std::string& file);


    /**
     * Stops those words whose weight is below minWeight.
     * Words are stopped by setting their weight to 0. There are not returned
     * later when transforming image features into vectors.
     * Note that when using IDF or TF_IDF, the weight is the idf part, which
     * is equivalent to -log(f), where f is the frequency of the word
     * (f = Ni/N, Ni: number of training images where the word is present,
     * N: number of training images).
     * Note that the old weight is forgotten, and subsequent calls to this
     * function with a lower minWeight have no effect.
     * @return number of words stopped now
     */
    virtual int stopWords(double minWeight);

   protected:
    /// Pointer to descriptor
    typedef const TDescriptor* pDescriptor;

    /// Tree node
    struct Node
    {
        /// Node id
        NodeId id;
        /// Weight if the node is a word
        WordValue weight;
        /// Children
        std::vector<NodeId> children;
        /// Parent node (undefined in case of root)
        NodeId parent;
        /// Node descriptor
        TDescriptor descriptor;

        /// Word id if the node is a word
        WordId word_id;

        /**
         * Empty constructor
         */
        Node() : id(0), weight(0), parent(0), word_id(0) {}

        /**
         * Constructor
         * @param _id node id
         */
        Node(NodeId _id) : id(_id), weight(0), parent(0), word_id(0) {}

        /**
         * Returns whether the node is a leaf node
         * @return true iff the node is a leaf
         */
        inline bool isLeaf() const { return children.empty(); }
    };

   protected:
    /**
     * Returns a set of pointers to descriptores
     * @param training_features all the features
     * @param features (out) pointers to the training features
     */
    void getFeatures(const std::vector<std::vector<TDescriptor>>& training_features,
                     std::vector<pDescriptor>& features) const;

    /**
     * Returns the word id associated to a feature
     * @param feature
     * @param id (out) word id
     * @param weight (out) word weight
     * @param nid (out) if given, id of the node "levelsup" levels up
     * @param levelsup
     */
    virtual void transform(const TDescriptor& feature, WordId& id, WordValue& weight, NodeId* nid = NULL,
                           int levelsup = 0) const;

    /**
     * Returns the word id associated to a feature
     * @param feature
     * @param id (out) word id
     */
    virtual void transform(const TDescriptor& feature, WordId& id) const;

    /**
     * Creates a level in the tree, under the parent, by running kmeans with
     * a descriptor set, and recursively creates the subsequent levels too
     * @param parent_id id of parent node
     * @param descriptors descriptors to run the kmeans on
     * @param current_level current level in the tree
     */
    void HKmeansStep(NodeId parent_id, const std::vector<pDescriptor>& descriptors, int current_level);

    /**
     * Creates k clusters from the given descriptors with some seeding algorithm.
     * @note In this class, kmeans++ is used, but this function should be
     *   overriden by inherited classes.
     */
    virtual void initiateClusters(const std::vector<pDescriptor>& descriptors,
                                  std::vector<TDescriptor>& clusters) const;

    /**
     * Creates k clusters from the given descriptor sets by running the
     * initial step of kmeans++
     * @param descriptors
     * @param clusters resulting clusters
     */
    void initiateClustersKMpp(const std::vector<pDescriptor>& descriptors, std::vector<TDescriptor>& clusters) const;

    /**
     * Create the words of the vocabulary once the tree has been built
     */
    void createWords();

    /**
     * Sets the weights of the nodes of tree according to the given features.
     * Before calling this function, the nodes and the words must be already
     * created (by calling HKmeansStep and createWords)
     * @param features
     */
    void setNodeWeights(const std::vector<std::vector<TDescriptor>>& features);

    /**
     * Returns a random number in the range [min..max]
     * @param min
     * @param max
     * @return random T number in [min..max]
     */
    template <class T>
    static T RandomValue(T min, T max)
    {
        return ((T)rand() / (T)RAND_MAX) * (max - min) + min;
    }

    /**
     * Returns a random int in the range [min..max]
     * @param min
     * @param max
     * @return random int in [min..max]
     */
    static int RandomInt(int min, int max)
    {
        int d = max - min + 1;
        return int(((double)rand() / ((double)RAND_MAX + 1.0)) * d) + min;
    }

   protected:
    /// Branching factor
    int m_k;

    /// Depth levels
    int m_L;

    /// Weighting method
    WeightingType m_weighting;



    /// Tree nodes
    std::vector<Node> m_nodes;

    /// Words of the vocabulary (tree leaves)
    /// this condition holds: m_words[wid]->word_id == wid
    std::vector<Node*> m_words;
};

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
TemplatedVocabulary<TDescriptor, F, Scoring>::TemplatedVocabulary(int k, int L, WeightingType weighting)
    : m_k(k), m_L(L), m_weighting(weighting)
{
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
TemplatedVocabulary<TDescriptor, F, Scoring>::TemplatedVocabulary(const std::string& filename)
{
    loadRaw(filename);
}


// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::setWeightingType(WeightingType type)
{
    this->m_weighting = type;
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
TemplatedVocabulary<TDescriptor, F, Scoring>::~TemplatedVocabulary()
{
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::create(
    const std::vector<std::vector<TDescriptor>>& training_features)
{
    m_nodes.clear();
    m_words.clear();

    // expected_nodes = Sum_{i=0..L} ( k^i )
    int expected_nodes = (int)((std::pow((double)m_k, (double)m_L + 1) - 1) / (m_k - 1));

    m_nodes.reserve(expected_nodes);  // avoid allocations when creating the tree


    std::vector<pDescriptor> features;
    getFeatures(training_features, features);


    // create root
    m_nodes.push_back(Node(0));  // root

    // create the tree
    HKmeansStep(0, features, 1);

    // create the words
    createWords();

    // and set the weight of each node of the tree
    setNodeWeights(training_features);
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::create(
    const std::vector<std::vector<TDescriptor>>& training_features, int k, int L)
{
    m_k = k;
    m_L = L;

    create(training_features);
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::create(
    const std::vector<std::vector<TDescriptor>>& training_features, int k, int L, WeightingType weighting)
{
    m_k         = k;
    m_L         = L;
    m_weighting = weighting;


    create(training_features);
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::getFeatures(
    const std::vector<std::vector<TDescriptor>>& training_features, std::vector<pDescriptor>& features) const
{
    features.resize(0);

    typename std::vector<std::vector<TDescriptor>>::const_iterator vvit;
    typename std::vector<TDescriptor>::const_iterator vit;
    for (vvit = training_features.begin(); vvit != training_features.end(); ++vvit)
    {
        features.reserve(features.size() + vvit->size());
        for (vit = vvit->begin(); vit != vvit->end(); ++vit)
        {
            features.push_back(&(*vit));
        }
    }
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::HKmeansStep(NodeId parent_id,
                                                               const std::vector<pDescriptor>& descriptors,
                                                               int current_level)
{
    if (descriptors.empty()) return;

    // features associated to each cluster
    std::vector<TDescriptor> clusters;
    std::vector<std::vector<unsigned int>> groups;  // groups[i] = [j1, j2, ...]
                                                    // j1, j2, ... indices of descriptors associated to cluster i

    clusters.reserve(m_k);
    groups.reserve(m_k);

    // const int msizes[] = { m_k, descriptors.size() };
    // cv::SparseMat assoc(2, msizes, CV_8U);
    // cv::SparseMat last_assoc(2, msizes, CV_8U);
    //// assoc.row(cluster_idx).col(descriptor_idx) = 1 iif associated

    if ((int)descriptors.size() <= m_k)
    {
        // trivial case: one cluster per feature
        groups.resize(descriptors.size());

        for (unsigned int i = 0; i < descriptors.size(); i++)
        {
            groups[i].push_back(i);

#ifdef USE_CV_FORB
            clusters.push_back(descriptors[i]->clone());
#else
            clusters.push_back(*descriptors[i]);
#endif
        }
    }
    else
    {
        // select clusters and groups with kmeans

        bool first_time = true;
        bool goon       = true;

        // to check if clusters move after iterations
        std::vector<int> last_association, current_association;

        while (goon)
        {
            // 1. Calculate clusters

            if (first_time)
            {
                // random sample
                initiateClusters(descriptors, clusters);
            }
            else
            {
                // calculate cluster centres

                for (unsigned int c = 0; c < clusters.size(); ++c)
                {
                    std::vector<pDescriptor> cluster_descriptors;
                    cluster_descriptors.reserve(groups[c].size());

                    /*
                    for(unsigned int d = 0; d < descriptors.size(); ++d)
                    {
                      if( assoc.find<unsigned char>(c, d) )
                      {
                        cluster_descriptors.push_back(descriptors[d]);
                      }
                    }
                    */

                    std::vector<unsigned int>::const_iterator vit;
                    for (vit = groups[c].begin(); vit != groups[c].end(); ++vit)
                    {
                        cluster_descriptors.push_back(descriptors[*vit]);
                    }


                    F::meanValue(cluster_descriptors, clusters[c]);
                }

            }  // if(!first_time)

            // 2. Associate features with clusters

            // calculate distances to cluster centers
            groups.clear();
            groups.resize(clusters.size(), std::vector<unsigned int>());
            current_association.resize(descriptors.size());

            // assoc.clear();

            typename std::vector<pDescriptor>::const_iterator fit;
            // unsigned int d = 0;
            for (fit = descriptors.begin(); fit != descriptors.end(); ++fit)  //, ++d)
            {
                double best_dist      = F::distance(*(*fit), clusters[0]);
                unsigned int icluster = 0;

                for (unsigned int c = 1; c < clusters.size(); ++c)
                {
                    double dist = F::distance(*(*fit), clusters[c]);
                    if (dist < best_dist)
                    {
                        best_dist = dist;
                        icluster  = c;
                    }
                }

                // assoc.ref<unsigned char>(icluster, d) = 1;

                groups[icluster].push_back(fit - descriptors.begin());
                current_association[fit - descriptors.begin()] = icluster;
            }

            // kmeans++ ensures all the clusters has any feature associated with them

            // 3. check convergence
            if (first_time)
            {
                first_time = false;
            }
            else
            {
                // goon = !eqUChar(last_assoc, assoc);

                goon = false;
                for (unsigned int i = 0; i < current_association.size(); i++)
                {
                    if (current_association[i] != last_association[i])
                    {
                        goon = true;
                        break;
                    }
                }
            }

            if (goon)
            {
                // copy last feature-cluster association
                last_association = current_association;
                // last_assoc = assoc.clone();
            }

        }  // while(goon)

    }  // if must run kmeans

    // create nodes
    for (unsigned int i = 0; i < clusters.size(); ++i)
    {
        NodeId id = m_nodes.size();
        m_nodes.push_back(Node(id));
        m_nodes.back().descriptor = clusters[i];
        m_nodes.back().parent     = parent_id;
        m_nodes[parent_id].children.push_back(id);
    }

    // go on with the next level
    if (current_level < m_L)
    {
        // iterate again with the resulting clusters
        const std::vector<NodeId>& children_ids = m_nodes[parent_id].children;
        for (unsigned int i = 0; i < clusters.size(); ++i)
        {
            NodeId id = children_ids[i];

            std::vector<pDescriptor> child_features;
            child_features.reserve(groups[i].size());

            std::vector<unsigned int>::const_iterator vit;
            for (vit = groups[i].begin(); vit != groups[i].end(); ++vit)
            {
                child_features.push_back(descriptors[*vit]);
            }

            if (child_features.size() > 1)
            {
                HKmeansStep(id, child_features, current_level + 1);
            }
        }
    }
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::initiateClusters(const std::vector<pDescriptor>& descriptors,
                                                                    std::vector<TDescriptor>& clusters) const
{
    initiateClustersKMpp(descriptors, clusters);
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::initiateClustersKMpp(const std::vector<pDescriptor>& pfeatures,
                                                                        std::vector<TDescriptor>& clusters) const
{
    // Implements kmeans++ seeding algorithm
    // Algorithm:
    // 1. Choose one center uniformly at random from among the data points.
    // 2. For each data point x, compute D(x), the distance between x and the nearest
    //    center that has already been chosen.
    // 3. Add one new data point as a center. Each point x is chosen with probability
    //    proportional to D(x)^2.
    // 4. Repeat Steps 2 and 3 until k centers have been chosen.
    // 5. Now that the initial centers have been chosen, proceed using standard k-means
    //    clustering.

    clusters.resize(0);
    clusters.reserve(m_k);
    std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());

    // 1.

    int ifeature = RandomInt(0, pfeatures.size() - 1);

// create first cluster
#ifdef USE_CV_FORB
    clusters.push_back(pfeatures[ifeature]->clone());
#else
    clusters.push_back(*pfeatures[ifeature]);
#endif

    // compute the initial distances
    typename std::vector<pDescriptor>::const_iterator fit;
    std::vector<double>::iterator dit;
    dit = min_dists.begin();
    for (fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
    {
        *dit = F::distance(*(*fit), clusters.back());
    }

    while ((int)clusters.size() < m_k)
    {
        // 2.
        dit = min_dists.begin();
        for (fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
        {
            if (*dit > 0)
            {
                double dist = F::distance(*(*fit), clusters.back());
                if (dist < *dit) *dit = dist;
            }
        }

        // 3.
        double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

        if (dist_sum > 0)
        {
            double cut_d;
            do
            {
                cut_d = RandomValue<double>(0, dist_sum);
            } while (cut_d == 0.0);

            double d_up_now = 0;
            for (dit = min_dists.begin(); dit != min_dists.end(); ++dit)
            {
                d_up_now += *dit;
                if (d_up_now >= cut_d) break;
            }

            if (dit == min_dists.end())
                ifeature = pfeatures.size() - 1;
            else
                ifeature = dit - min_dists.begin();

#ifdef USE_CV_FORB
            clusters.push_back(pfeatures[ifeature]->clone());
#else
            clusters.push_back(*pfeatures[ifeature]);
#endif

        }  // if dist_sum > 0
        else
            break;

    }  // while(used_clusters < m_k)
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::createWords()
{
    m_words.resize(0);

    if (!m_nodes.empty())
    {
        m_words.reserve((int)pow((double)m_k, (double)m_L));

        typename std::vector<Node>::iterator nit;

        nit = m_nodes.begin();  // ignore root
        for (++nit; nit != m_nodes.end(); ++nit)
        {
            if (nit->isLeaf())
            {
                nit->word_id = m_words.size();
                m_words.push_back(&(*nit));
            }
        }
    }
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::setNodeWeights(
    const std::vector<std::vector<TDescriptor>>& training_features)
{
    const unsigned int NWords = m_words.size();
    const unsigned int NDocs  = training_features.size();

    if (m_weighting == TF || m_weighting == BINARY)
    {
        // idf part must be 1 always
        for (unsigned int i = 0; i < NWords; i++) m_words[i]->weight = 1;
    }
    else if (m_weighting == IDF || m_weighting == TF_IDF)
    {
        // IDF and TF-IDF: we calculte the idf path now

        // Note: this actually calculates the idf part of the tf-idf score.
        // The complete tf-idf score is calculated in ::transform

        std::vector<unsigned int> Ni(NWords, 0);
        std::vector<bool> counted(NWords, false);

        typename std::vector<std::vector<TDescriptor>>::const_iterator mit;
        typename std::vector<TDescriptor>::const_iterator fit;

        for (mit = training_features.begin(); mit != training_features.end(); ++mit)
        {
            fill(counted.begin(), counted.end(), false);

            for (fit = mit->begin(); fit < mit->end(); ++fit)
            {
                WordId word_id;
                transform(*fit, word_id);

                if (!counted[word_id])
                {
                    Ni[word_id]++;
                    counted[word_id] = true;
                }
            }
        }

        // set ln(N/Ni)
        for (unsigned int i = 0; i < NWords; i++)
        {
            if (Ni[i] > 0)
            {
                m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
            }  // else // This cannot occur if using kmeans++
        }
    }
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
inline unsigned int TemplatedVocabulary<TDescriptor, F, Scoring>::size() const
{
    return m_words.size();
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
inline bool TemplatedVocabulary<TDescriptor, F, Scoring>::empty() const
{
    return m_words.empty();
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
float TemplatedVocabulary<TDescriptor, F, Scoring>::getEffectiveLevels() const
{
    long sum = 0;
    typename std::vector<Node*>::const_iterator wit;
    for (wit = m_words.begin(); wit != m_words.end(); ++wit)
    {
        const Node* p = *wit;

        for (; p->id != 0; sum++) p = &m_nodes[p->parent];
    }

    return (float)((double)sum / (double)m_words.size());
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
TDescriptor TemplatedVocabulary<TDescriptor, F, Scoring>::getWord(WordId wid) const
{
    return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
WordValue TemplatedVocabulary<TDescriptor, F, Scoring>::getWordWeight(WordId wid) const
{
    return m_words[wid]->weight;
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
WordId TemplatedVocabulary<TDescriptor, F, Scoring>::transform(const TDescriptor& feature) const
{
    if (empty())
    {
        return 0;
    }

    WordId wid;
    transform(feature, wid);
    return wid;
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::transform(const std::vector<TDescriptor>& features,
                                                             BowVector& v) const
{
    v.clear();

    if (empty())
    {
        return;
    }

    // normalize
    //    LNorm norm;
    //    bool must = m_scoring_object->mustNormalize(norm);

    typename std::vector<TDescriptor>::const_iterator fit;

    if (m_weighting == TF || m_weighting == TF_IDF)
    {
        for (fit = features.begin(); fit < features.end(); ++fit)
        {
            WordId id;
            WordValue w;
            // w is the idf value if TF_IDF, 1 if TF

            transform(*fit, id, w);

            // not stopped
            if (w > 0) v.addWeight(id, w);
        }

        if (!v.empty() && !Scoring::mustNormalize)
        {
            // unnecessary when normalizing
            const double nd = v.size();
            for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
        }
    }
    else  // IDF || BINARY
    {
        for (fit = features.begin(); fit < features.end(); ++fit)
        {
            WordId id;
            WordValue w;
            // w is idf if IDF, or 1 if BINARY

            transform(*fit, id, w);

            // not stopped
            if (w > 0) v.addIfNotExist(id, w);

        }  // if add_features
    }      // if m_weighting == ...

    if (Scoring::mustNormalize) v.normalize();
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::transform(const std::vector<TDescriptor>& features, BowVector& v,
                                                             FeatureVector& fv, int levelsup) const
{
    int N                 = features.size();
    using TransformResult = std::tuple<WordId, NodeId, WordValue>;
    std::vector<TransformResult> transformedFeatures(N);

    v.clear();
    fv.clear();


    if (empty())  // safe for subclasses
    {
        return;
    }



    if (m_weighting == TF || m_weighting == TF_IDF)
    {
        for (int i = 0; i < N; ++i)
        {
            WordId& id   = std::get<0>(transformedFeatures[i]);
            NodeId& nid  = std::get<1>(transformedFeatures[i]);
            WordValue& w = std::get<2>(transformedFeatures[i]);
            // w is the idf value if TF_IDF, 1 if TF

            transform(features[i], id, w, &nid, levelsup);
        }

        for (int i = 0; i < N; ++i)
        {
            WordId& id   = std::get<0>(transformedFeatures[i]);
            NodeId& nid  = std::get<1>(transformedFeatures[i]);
            WordValue& w = std::get<2>(transformedFeatures[i]);

            if (w > 0)  // not stopped
            {
                v.addWeight(id, w);
                fv.addFeature(nid, i);
            }
        }


        if (!v.empty() && !Scoring::mustNormalize)
        {
            // unnecessary when normalizing
            const double nd = v.size();
            for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
        }
    }
    else  // IDF || BINARY
    {
        typename std::vector<TDescriptor>::const_iterator fit;
        throw std::runtime_error("not supported");
        unsigned int i_feature = 0;
        for (fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
        {
            WordId id;
            NodeId nid;
            WordValue w;
            // w is idf if IDF, or 1 if BINARY

            transform(*fit, id, w, &nid, levelsup);

            if (w > 0)  // not stopped
            {
                v.addIfNotExist(id, w);
                fv.addFeature(nid, i_feature);
            }
        }
    }  // if m_weighting == ...

    if (Scoring::mustNormalize) v.normalize();
}


template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::transformOMP(const std::vector<TDescriptor>& features, BowVector& v,
                                                                FeatureVector& fv, int levelsup)
{
#pragma omp single
    {
        N = features.size();
        transformedFeatures.resize(N);

        v.clear();
        fv.clear();
    }

    if (empty())  // safe for subclasses
    {
        return;
    }



#pragma omp for
    for (int i = 0; i < N; ++i)
    {
        WordId& id   = std::get<0>(transformedFeatures[i]);
        NodeId& nid  = std::get<1>(transformedFeatures[i]);
        WordValue& w = std::get<2>(transformedFeatures[i]);
        // w is the idf value if TF_IDF, 1 if TF

        transform(features[i], id, w, &nid, levelsup);
    }
#pragma omp single
    {
        for (int i = 0; i < N; ++i)
        {
            WordId& id   = std::get<0>(transformedFeatures[i]);
            NodeId& nid  = std::get<1>(transformedFeatures[i]);
            WordValue& w = std::get<2>(transformedFeatures[i]);

            if (w > 0)  // not stopped
            {
                v.addWeight(id, w);
                fv.addFeature(nid, i);
            }
        }


        if (!v.empty() && !Scoring::mustNormalize)
        {
            // unnecessary when normalizing
            const double nd = v.size();
            for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
        }



        if (Scoring::mustNormalize) v.normalize();
    }
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
inline double TemplatedVocabulary<TDescriptor, F, Scoring>::score(const BowVector& v1, const BowVector& v2) const
{
    return Scoring::score(v1, v2);
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::transform(const TDescriptor& feature, WordId& id) const
{
    WordValue weight;
    transform(feature, id, weight);
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::transform(const TDescriptor& feature, WordId& word_id,
                                                             WordValue& weight, NodeId* nid, int levelsup) const
{
    // propagate the feature down the tree
    //    std::vector<NodeId> nodes;
    //    typename std::vector<NodeId>::const_iterator nit;

    // level at which the node must be stored in nid, if given
    const int nid_level = m_L - levelsup;
    if (nid_level <= 0 && nid != NULL) *nid = 0;  // root

    NodeId final_id   = 0;  // root
    int current_level = 0;

    do
    {
        ++current_level;
        auto& nodes = m_nodes[final_id].children;
        final_id    = nodes[0];

        double best_d = F::distance(feature, m_nodes[final_id].descriptor);

        for (auto nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
        {
            NodeId id = *nit;
            double d  = F::distance(feature, m_nodes[id].descriptor);
            if (d < best_d)
            {
                best_d   = d;
                final_id = id;
            }
        }

        if (nid != NULL && current_level == nid_level) *nid = final_id;

    } while (!m_nodes[final_id].isLeaf());

    // turn node id into word id
    word_id = m_nodes[final_id].word_id;
    weight  = m_nodes[final_id].weight;
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
NodeId TemplatedVocabulary<TDescriptor, F, Scoring>::getParentNode(WordId wid, int levelsup) const
{
    NodeId ret = m_words[wid]->id;    // node id
    while (levelsup > 0 && ret != 0)  // ret == 0 --> root
    {
        --levelsup;
        ret = m_nodes[ret].parent;
    }
    return ret;
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::getWordsFromNode(NodeId nid, std::vector<WordId>& words) const
{
    words.clear();

    if (m_nodes[nid].isLeaf())
    {
        words.push_back(m_nodes[nid].word_id);
    }
    else
    {
        words.reserve(m_k);  // ^1, ^2, ...

        std::vector<NodeId> parents;
        parents.push_back(nid);

        while (!parents.empty())
        {
            NodeId parentid = parents.back();
            parents.pop_back();

            const std::vector<NodeId>& child_ids = m_nodes[parentid].children;
            std::vector<NodeId>::const_iterator cit;

            for (cit = child_ids.begin(); cit != child_ids.end(); ++cit)
            {
                const Node& child_node = m_nodes[*cit];

                if (child_node.isLeaf())
                    words.push_back(child_node.word_id);
                else
                    parents.push_back(*cit);

            }  // for each child
        }      // while !parents.empty
    }
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F, class Scoring>
int TemplatedVocabulary<TDescriptor, F, Scoring>::stopWords(double minWeight)
{
    int c = 0;
    typename std::vector<Node*>::iterator wit;
    for (wit = m_words.begin(); wit != m_words.end(); ++wit)
    {
        if ((*wit)->weight < minWeight)
        {
            ++c;
            (*wit)->weight = 0;
        }
    }
    return c;
}


struct BinaryFile
{
    BinaryFile(const std::string& file, std::ios_base::openmode __mode = std::ios_base::in)
        : strm(file, std::ios::binary | __mode)
    {
    }

    template <typename T>
    void write(const T& v)
    {
        strm.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }

    template <typename T>
    void write(const std::vector<T>& vec)
    {
        write((size_t)vec.size());
        for (auto& v : vec) write(v);
    }

    template <typename T>
    void read(std::vector<T>& vec)
    {
        size_t s;
        read(s);
        vec.resize(s);
        for (auto& v : vec) read(v);
    }

    template <typename T>
    void read(T& v)
    {
        strm.read(reinterpret_cast<char*>(&v), sizeof(T));
    }

    template <typename T>
    BinaryFile& operator<<(const T& v)
    {
        write(v);
        return *this;
    }

    template <typename T>
    BinaryFile& operator>>(T& v)
    {
        read(v);
        return *this;
    }

    std::fstream strm;
};

template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::loadRaw(const std::string& file)
{
    BinaryFile bf(file, std::ios_base::in);
    if (!bf.strm.is_open())
    {
        throw std::runtime_error("Could not load Voc file.");
    }
    int scoringid;
    bf >> m_k >> m_L >> scoringid >> m_weighting;

    if (m_weighting != TF_IDF)
    {
        throw std::runtime_error("Only TF_IDF supported.");
    }
    if (scoringid != Scoring::id)
    {
        throw std::runtime_error("Scoring id doesn't match template.");
    }


    size_t nodecount;
    bf >> nodecount;
    m_nodes.resize(nodecount);
    for (Node& n : m_nodes)
    {
        bf >> n.id >> n.parent >> n.weight >> n.word_id >> n.descriptor;
        if (n.id != 0) m_nodes[n.parent].children.push_back(n.id);
    }

    // words
    std::vector<std::pair<int, int>> words;
    bf >> words;

    m_words.resize(words.size());
    for (auto i = 0; i < m_words.size(); ++i)
    {
        m_words[i] = &m_nodes[words[i].second];
    }
}



template <class TDescriptor, class F, class Scoring>
void TemplatedVocabulary<TDescriptor, F, Scoring>::saveRaw(const std::string& file) const
{
    BinaryFile bf(file, std::ios_base::out);
    bf << m_k << m_L << Scoring::id << m_weighting;
    bf << (size_t)m_nodes.size();
    for (const Node& n : m_nodes)
    {
        bf << n.id << n.parent << n.weight << n.word_id << n.descriptor;
    }
    // words
    std::vector<std::pair<int, int>> words;
    for (auto i = 0; i < (int)m_words.size(); ++i)
    {
        words.emplace_back(i, m_words[i]->id);
    }
    bf << words;
}



// --------------------------------------------------------------------------

/**
 * Writes printable information of the vocabulary
 * @param os stream to write to
 * @param voc
 */
template <class TDescriptor, class F, class Scoring>
std::ostream& operator<<(std::ostream& os, const TemplatedVocabulary<TDescriptor, F, Scoring>& voc)
{
    os << "Vocabulary: k = " << voc.getBranchingFactor() << ", L = " << voc.getDepthLevels() << ", Weighting = ";

    switch (voc.getWeightingType())
    {
        case TF_IDF:
            os << "tf-idf";
            break;
        case TF:
            os << "tf";
            break;
        case IDF:
            os << "idf";
            break;
        case BINARY:
            os << "binary";
            break;
    }

    os << ", Scoring = ";
    switch (Scoring::id)
    {
        case 0:
            os << "L1-norm";
            break;
    }

    os << ", Number of words = " << voc.size();

    return os;
}

}  // namespace MiniBow
