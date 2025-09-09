import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    #raise NotImplementedError
    # 1. determine if the person is a parent or a child
    
    joint_prob = 1
    
    for person in people:
        
        mom = people[person]['mother']
        dad = people[person]['father']
        
        people_prob=1
        people_gene = 2 if person in two_genes else 1 if person in one_gene else 0
        people_trait = True if person in have_trait else False
    
        if not mom and not dad:
            people_prob *= PROBS['gene'][people_gene]
            
        else:
            # for mom
            if mom in two_genes:
                mom_prob =1 - PROBS['mutation']
            elif mom in one_gene:
                mom_prob =0.5
            else:
                mom_prob =PROBS['mutation']

            #for dad
            if dad in two_genes:
                dad_prob =  1 - PROBS['mutation']
            elif dad in one_gene:
                dad_prob =  0.5
            else:
                dad_prob =  PROBS['mutation']
            # for children
            
            if people_gene == 2:
                people_prob *= mom_prob * dad_prob
            elif people_gene == 1:
                people_prob *= (1 - mom_prob) * dad_prob + (1 - dad_prob) * mom_prob
            else:
                people_prob *= (1 - mom_prob) * (1 - dad_prob)   

        people_prob *= PROBS['trait'][people_gene][people_trait]

        joint_prob *= people_prob
    
    return joint_prob
    """child = set()
    parent = set()
    probablity = set()
    joint_prob = 1
    joint_prob_parent = 1
    joint_prob_child = 1
    
    mutation = PROBS["mutation"]
    
    for person in people :
        if not people[person]['mother'] and not people[person]['father']:
            parent.add(person)
        else:
            child.add(person)
    
    # for parent and actually for individual 
    for person_parent in parent:
        gene = 2 if person_parent in two_genes else 1 if person_parent in one_gene else 0
        parent_trait = True if person_parent in have_trait else False
        joint_prob_parent *= PROBS["gene"][gene] * PROBS["trait"][gene][parent_trait]
        
    
    for person_child in child:
    
        # 0. who's this child's parent?
        father = people[person_child]["father"]
        father_gene = 2 if father in two_genes else 1 if father in one_gene else 0
        #father_trait = True if person_parent in have_trait else False
        mother = people[person_child]["mother"]
        mother_gene = 2 if mother in two_genes else 1 if mother in one_gene else 0
        #mother_trait =  True if person_parent in have_trait else False
    
        child_gene = 2 if person_child in two_genes else 1 if person_child in one_gene else 0
        child_trait = True if person_child in have_trait else False
        # 1. if child has zero gene
        #havn't deal with trait yet
        if child_gene ==0:
            if father_gene + mother_gene ==0:
               joint_prob_child = (1-mutation)*(1-mutation)
            if father_gene + mother_gene ==1:
               joint_prob_child = (1-mutation)*0.5
            if father_gene + mother_gene ==2:
               if father_gene==mother_gene:
                  joint_prob_child = 0.5*0.5
               else:
                  joint_prob_child = mutation*(1-mutation)
            if father_gene + mother_gene ==3:
               joint_prob_child = mutation*0.5
            if father_gene + mother_gene ==4:
               joint_prob_child = mutation*mutation
        if child_gene ==1:
            if father_gene + mother_gene ==0 or father_gene + mother_gene ==4:
               joint_prob_child = (1-mutation)  *mutation
            if father_gene + mother_gene ==1 or father_gene + mother_gene ==3:
               joint_prob_child = (1-mutation) *0.5+ mutation *0.5
            if father_gene + mother_gene ==2:
                if father_gene==mother_gene:
                    joint_prob_child = 0.5*0.5
                else:
                    joint_prob_child = 0.5*0.5+(1-mutation)*(1-mutation)
        if child_gene == 2:
            if father_gene + mother_gene ==4:
               joint_prob_child = (1-mutation)*(1-mutation)
            if father_gene + mother_gene ==3:
               joint_prob_child = (1-mutation)*0.5
            if father_gene + mother_gene ==2:
               if father_gene==mother_gene:
                  joint_prob_child = 0.5*0.5
               else:
                  joint_prob_child = mutation*(1-mutation)
            if father_gene + mother_gene ==1:
               joint_prob_child = mutation*0.5
            if father_gene + mother_gene ==0:
               joint_prob_child = mutation*mutation
            
        joint_prob_child *= PROBS["trait"][child_gene][child_trait]
    
    joint_prob = joint_prob_parent*joint_prob_child"""
    


    


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    #raise NotImplementedError
    for person in probabilities:
        person_genes = (2 if person in two_genes else 1 if person in one_gene else 0)
        person_trait = person in have_trait

    # Update person probability distributions for gene and trait
        probabilities[person]['gene'][person_genes] += p
        probabilities[person]['trait'][person_trait] += p
    
    
    
    


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    #raise NotImplementedError
    for person in probabilities:

        # Calculate the total probability for each distribution
        gene_prob_sum = sum(probabilities[person]['gene'].values())
        trait_prob_sum = sum(probabilities[person]['trait'].values())

        # Normalise each distribution to 1:
        probabilities[person]['gene'] = { genes: (prob / gene_prob_sum) for genes, prob in probabilities[person]['gene'].items()}
        probabilities[person]['trait'] = { trait: (prob / trait_prob_sum) for trait, prob in probabilities[person]['trait'].items()}



if __name__ == "__main__":
    main()

class MyList(list):
    def __init__(self, *args):
        super(MyList, self).__init__(args)

    def __sub__(self, other):
        return self.__class__(*[item for item in self if item not in other])