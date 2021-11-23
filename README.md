# Asym

## Introduction 
Asym helps handle asymmetric pytorch data (e.g. those with variable length) easily. 
Aside from simple features like

* batching (with padding)
* unbatching (with unpadding)

, asym also provides sophisticated features such as  

* grouping (explained in the following section)



--------------------------------

## Unique features 
### Unbatching after module application  

Unbatching a batched data after applying a torch.nn.Module cannot be done without additional information, since it could be unclear which part of the tensors should be unpadded. Asym achieves this by requiring the user to write a shallow wrapper class of the module (inherit from *AnnotatedModule*) that annotates the input and output *dimension types*. This allows unpadding only those parts of the output tensors that correspond to the padded length index of the input tensors. 

### Unusual padding methods

With the help of dimension type annotations, some unusual (and sometimes useful) padding methods can be easily denoted by users: 
* CDimPadder : Pad such that "constant dimensions" (jointly) take a given tensor value at padded positions. 

### Grouping

Suppose you have a vastly asymmetric set of data (Let's say some have sequence length 1000 while others have sequence length <10), and you want to pass them to a module (torch.nn.Module) and get the results of module.forward() application. Convential ways of doing that would include:

1. Processing them one by one, in which case we cannot exploit the device(e.g. GPU)'s full capacity.
2. Processing them in mini-batches, which may require gross amounts of padding if you plainly do it.

A solution for this dilemma is to group the set in such a way that each group contains only the data with similar lengths, process the resulting mini-batches, and get the result back. 
Asym allow users to write custom *Grouper* classes for this process.

## Examples
Run
```
. prep
python3 examples.py
```
(later, simple examples will be here in README)

## Current limitations 

* A *length dimension* cannot be introduced in an AnnotatedModule application. This means, for example, after applying a CNN layer that shrinks image width and height, data cannot be properly unbatched. (Since you don't know which part along the shrinked length dimension should be unpadded) This will be fixed soon. 