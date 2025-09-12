from nltk.corpus import wordnet

def get_parents(synset):
    """Get the parent synsets of a given synset."""
    return synset.hypernyms()

def get_children(synset):
    """Get the child synsets of a given synset."""
    return synset.hyponyms()

def get_siblings(synset):
    """Get the sibling synsets of a given synset."""
    parents = get_parents(synset)
    siblings = set()
    for parent in parents:
        siblings.update(get_children(parent))
    siblings.discard(synset)  # Remove the original synset from its siblings
    return list(siblings)

def get_synset(word, pos=None):
    """Get the synset for a given word and part of speech."""
    synsets = wordnet.synsets(word, pos=pos)
    return synsets[0] if synsets else None

print(get_synset("quick", pos='a'))