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

#include "saiga/core/time/all.h"
#include "saiga/core/util/BinaryFile.h"
#include "saiga/vision/features/Features.h"

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

namespace MiniBow2
{
using WordId     = int;
using WordValue  = float;
using NodeId     = int;
using Descriptor = Saiga::DescriptorORB;

class BowVector : public std::vector<std::pair<WordId, WordValue>>
// class BowVector : public std::map<WordId, WordValue>
{
   public:
    void set(std::vector<std::pair<WordId, WordValue>>& words)
    {
#if 1
        reserve(words.size());


        std::sort(words.begin(), words.end(), [](const auto& a, const auto& b) { return a.first < b.first; });


        WordId current = -1;
        for (auto& f : words)
        {
            //            SAIGA_ASSERT(f.first >= 0);
            if (f.first == -1) continue;

            if (f.first != current)
            {
                push_back({f.first, f.second});
                current = f.first;
            }
            else
            {
                back().second += (f.second);
            }
        }
#else
        for (auto& f : words)
        {
            this->operator[](f.first) += f.second;
        }
#endif
        normalize();
    }

    /**
     * L1-Normalizes the values in the vector
     * @param norm_type norm used
     */
    void normalize()
    {
        WordValue norm = 0.0;
        {
            for (auto it = begin(); it != end(); ++it) norm += std::abs(it->second);
        }
        if (norm > 0.0)
        {
            for (auto it = begin(); it != end(); ++it) it->second /= norm;
        }
    }
};

class FeatureVector : public std::vector<std::pair<NodeId, std::vector<int>>>
// class FeatureVector : public std::map<NodeId, std::vector<int>>
{
   public:
    void setFeatures(std::vector<std::pair<NodeId, int>>& features)
    {
#if 1
        reserve(features.size());

        std::sort(features.begin(), features.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        NodeId current = -1;
        for (auto& f : features)
        {
            if (f.first == -1) continue;
            SAIGA_ASSERT(f.first >= 0);



            if (f.first != current)
            {
                push_back({f.first, { f.second }});
                current = f.first;
            }
            else
            {
                back().second.push_back(f.second);
            }
        }
#else

        for (auto& f : features)
        {
            this->operator[](f.first).push_back(f.second);
        }
#endif
    }

    static WordValue score(const BowVector& v1, const BowVector& v2)
    {
        BowVector::const_iterator v1_it, v2_it;
        const BowVector::const_iterator v1_end = v1.end();
        const BowVector::const_iterator v2_end = v2.end();
        v1_it                                  = v1.begin();
        v2_it                                  = v2.begin();
        WordValue score                        = 0;
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
                v1_it = std::lower_bound(v1_it, v1.end(), *v2_it,
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            else
            {
                v2_it = std::lower_bound(v2_it, v2.end(), *v1_it,
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
            }
        }
        score = score * WordValue(-0.5);

        return score;
    }
};



template <class Descriptor>
class TemplatedVocabulary
{
   public:
    using Scoring = FeatureVector;
    /**
     * Initiates an empty vocabulary
     * @param k branching factor
     * @param L depth levels
     * @param weighting weighting type
     * @param scoring scoring type
     */
    TemplatedVocabulary(int k = 10, int L = 5) : m_k(k), m_L(L) {}

    /**
     * Creates the vocabulary by loading a file
     * @param filename
     */
    TemplatedVocabulary(const std::string& filename) { loadRaw(filename); }

    /**
     * Creates a vocabulary from the training features with the already
     * defined parameters
     * @param training_features
     */
    void create(const std::vector<std::vector<Descriptor>>& training_features);

    /**
     * Creates a vocabulary from the training features, setting the branching
     * factor and the depth levels of the tree
     * @param training_features
     * @param k branching factor
     * @param L depth levels
     */
    void create(const std::vector<std::vector<Descriptor>>& training_features, int k, int L)
    {
        m_k = k;
        m_L = L;
        create(training_features);
    }

    /**
     * Returns the number of words in the vocabulary
     * @return number of words
     */
    inline unsigned int size() const { return m_words.size(); }

    /**
     * Returns whether the vocabulary is empty (i.e. it has not been trained)
     * @return true iff the vocabulary is empty
     */
    inline bool empty() const { return m_words.empty(); }

    /**
     * Transforms a set of descriptores into a bow vector
     * @param features
     * @param v (out) bow vector of weighted words
     */
    //     void transform(const std::vector<TDescriptor>& features, BowVector& v) const;

    /**
     * Transform a set of descriptors into a bow vector and a feature vector
     * @param features
     * @param v (out) bow vector
     * @param fv (out) feature vector of nodes and feature indexes
     * @param levelsup levels to go up the vocabulary tree to get the node index
     */
    void transform(const std::vector<Descriptor>& features, BowVector& v, FeatureVector& fv, int levelsup,
                   int num_threads = 1) const;


    /**
     * Returns the score of two vectors
     * @param a vector
     * @param b vector
     * @return score between vectors
     * @note the vectors must be already sorted and normalized if necessary
     */
    inline WordValue score(const BowVector& a, const BowVector& b) const { return Scoring::score(a, b); }

    /**
     * Returns the id of the node that is "levelsup" levels from the word given
     * @param wid word id
     * @param levelsup 0..L
     * @return node id. if levelsup is 0, returns the node id associated to the
     *   word id
     */
    NodeId getParentNode(WordId wid, int levelsup) const;

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
    inline Descriptor getWord(WordId wid) const { return m_words[wid]->descriptor; }

    /**
     * Returns the weight of a word
     * @param wid word id
     * @return weight
     */
    inline WordValue getWordWeight(WordId wid) const { return m_words[wid]->weight; }

    /**
     * Changes the scoring method
     * @param type new scoring type
     */



    void saveRaw(const std::string& file) const;
    void loadRaw(const std::string& file);



   protected:
    /// Pointer to descriptor
    typedef const Descriptor* pDescriptor;

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
        Descriptor descriptor;

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
    void getFeatures(const std::vector<std::vector<Descriptor>>& training_features,
                     std::vector<pDescriptor>& features) const;

    /**
     * Returns the word id associated to a feature
     * @param feature
     * @param id (out) word id
     * @param weight (out) word weight
     * @param nid (out) if given, id of the node "levelsup" levels up
     * @param levelsup
     */
    std::tuple<WordId, WordValue, NodeId> transform(const Descriptor& feature, int levelsup) const;


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
    void initiateClusters(const std::vector<pDescriptor>& descriptors, std::vector<Descriptor>& clusters) const
    {
        initiateClustersKMpp(descriptors, clusters);
    }

    /**
     * Creates k clusters from the given descriptor sets by running the
     * initial step of kmeans++
     * @param descriptors
     * @param clusters resulting clusters
     */
    void initiateClustersKMpp(const std::vector<pDescriptor>& descriptors, std::vector<Descriptor>& clusters) const;

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
    void setNodeWeights(const std::vector<std::vector<Descriptor>>& features);

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

    /// Tree nodes
    std::vector<Node> m_nodes;

    /// Words of the vocabulary (tree leaves)
    /// this condition holds: m_words[wid]->word_id == wid
    std::vector<Node*> m_words;


    mutable std::vector<std::pair<WordId, WordValue>> tmp_bow_data;
    mutable std::vector<std::pair<NodeId, int>> tmp_feature_data;
};


// --------------------------------------------------------------------------

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::create(const std::vector<std::vector<Descriptor>>& training_features)
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

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::getFeatures(const std::vector<std::vector<Descriptor>>& training_features,
                                                  std::vector<pDescriptor>& features) const
{
    features.resize(0);

    typename std::vector<std::vector<Descriptor>>::const_iterator vvit;
    typename std::vector<Descriptor>::const_iterator vit;
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

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::HKmeansStep(NodeId parent_id, const std::vector<pDescriptor>& descriptors,
                                                  int current_level)
{
    if (descriptors.empty()) return;

    // features associated to each cluster
    std::vector<Descriptor> clusters;
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


                    Saiga::MeanMatcher<Descriptor> mm;
                    clusters[c] = mm.MeanDescriptorp(cluster_descriptors);
                    //                    clusters[c] = F::meanValue(cluster_descriptors);
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
                auto best_dist        = Saiga::distance(*(*fit), clusters[0]);
                unsigned int icluster = 0;

                for (unsigned int c = 1; c < clusters.size(); ++c)
                {
                    auto dist = Saiga::distance(*(*fit), clusters[c]);
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

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::initiateClustersKMpp(const std::vector<pDescriptor>& pfeatures,
                                                           std::vector<Descriptor>& clusters) const
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
        *dit = Saiga::distance(*(*fit), clusters.back());
    }

    while ((int)clusters.size() < m_k)
    {
        // 2.
        dit = min_dists.begin();
        for (fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
        {
            if (*dit > 0)
            {
                auto dist = Saiga::distance(*(*fit), clusters.back());
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

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::createWords()
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

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::setNodeWeights(const std::vector<std::vector<Descriptor>>& training_features)
{
    const unsigned int NWords = m_words.size();
    const unsigned int NDocs  = training_features.size();


    // IDF and TF-IDF: we calculte the idf path now

    // Note: this actually calculates the idf part of the tf-idf score.
    // The complete tf-idf score is calculated in ::transform

    std::vector<unsigned int> Ni(NWords, 0);
    std::vector<bool> counted(NWords, false);

    typename std::vector<std::vector<Descriptor>>::const_iterator mit;
    typename std::vector<Descriptor>::const_iterator fit;

    for (mit = training_features.begin(); mit != training_features.end(); ++mit)
    {
        fill(counted.begin(), counted.end(), false);

        for (fit = mit->begin(); fit < mit->end(); ++fit)
        {
            WordId word_id = std::get<0>(transform(*fit, 0));


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


// --------------------------------------------------------------------------

template <class Descriptor>
float TemplatedVocabulary<Descriptor>::getEffectiveLevels() const
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

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::transform(const std::vector<Descriptor>& features, BowVector& v,
                                                FeatureVector& fv, int levelsup, int num_threads) const
{
    SAIGA_ASSERT(num_threads > 0);
    int N = features.size();

    tmp_bow_data.resize(N);
    tmp_feature_data.resize(N);

    v.clear();
    fv.clear();

#pragma omp parallel num_threads(num_threads)
    {
#pragma omp for
        for (int i = 0; i < N; ++i)
        {
            auto [word_id, weight, nid] = transform(features[i], levelsup);

            if (weight > 0)
            {
                tmp_bow_data[i]     = {word_id, weight};
                tmp_feature_data[i] = {nid, i};
            }
            else
            {
                tmp_bow_data[i]     = {-1, weight};
                tmp_feature_data[i] = {-1, i};
            }
        }


#ifdef WIN32
#    pragma omp single
        {
            v.set(tmp_bow_data);
            fv.setFeatures(tmp_feature_data);
        }
#else
#    pragma omp single
        {
#    pragma omp task
            {
                v.set(tmp_bow_data);
            }

            fv.setFeatures(tmp_feature_data);
        }
#endif
    }
}

// --------------------------------------------------------------------------

template <class Descriptor>
std::tuple<WordId, WordValue, NodeId> TemplatedVocabulary<Descriptor>::transform(const Descriptor& feature,
                                                                                 int levelsup) const
{
    // propagate the feature down the tree
    //    std::vector<NodeId> nodes;
    //    typename std::vector<NodeId>::const_iterator nit;

    // level at which the node must be stored in nid, if given
    const int nid_level = m_L - levelsup;

    NodeId nid = 0;

    NodeId final_id   = 0;  // root
    int current_level = 0;

    do
    {
        ++current_level;
        auto& nodes = m_nodes[final_id].children;
        final_id    = nodes[0];

        auto best_d = Saiga::distance(feature, m_nodes[final_id].descriptor);

        for (auto nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
        {
            NodeId id = *nit;
            auto d    = Saiga::distance(feature, m_nodes[id].descriptor);
            if (d < best_d)
            {
                best_d   = d;
                final_id = id;
            }
        }

        if (current_level == nid_level) nid = final_id;

    } while (!m_nodes[final_id].isLeaf());

    // turn node id into word id
    WordId word_id   = m_nodes[final_id].word_id;
    WordValue weight = m_nodes[final_id].weight;

    return {word_id, weight, nid};
}

// --------------------------------------------------------------------------

template <class Descriptor>
NodeId TemplatedVocabulary<Descriptor>::getParentNode(WordId wid, int levelsup) const
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

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::getWordsFromNode(NodeId nid, std::vector<WordId>& words) const
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

template <class Descriptor>
void TemplatedVocabulary<Descriptor>::loadRaw(const std::string& file)
{
    Saiga::BinaryFile bf(file, std::ios_base::in);
    if (!bf.strm.is_open())
    {
        throw std::runtime_error("Could not load Voc file.");
    }
    int scoringid;
    int m_weighting_old;
    bf >> m_k >> m_L >> scoringid >> m_weighting_old;



    size_t nodecount;
    bf >> nodecount;
    m_nodes.resize(nodecount);
    for (Node& n : m_nodes)
    {
        double weight;
        bf >> n.id >> n.parent >> weight >> n.word_id >> n.descriptor;
        n.weight = weight;
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



template <class Descriptor>
void TemplatedVocabulary<Descriptor>::saveRaw(const std::string& file) const
{
    Saiga::BinaryFile bf(file, std::ios_base::out);
    bf << m_k << m_L << int(0) << int(0);
    bf << (size_t)m_nodes.size();
    for (const Node& n : m_nodes)
    {
        double weight = n.weight;
        bf << n.id << n.parent << weight << n.word_id << n.descriptor;
    }
    // words
    std::vector<std::pair<int, int>> words;
    for (auto i = 0; i < m_words.size(); ++i)
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
template <class Descriptor>
std::ostream& operator<<(std::ostream& os, const TemplatedVocabulary<Descriptor>& voc)
{
    os << "Vocabulary: k = " << voc.getBranchingFactor() << ", L = " << voc.getDepthLevels() << ", Weighting = ";

    os << ", Scoring = ";

    os << ", Number of words = " << voc.size();

    return os;
}

}  // namespace MiniBow2
