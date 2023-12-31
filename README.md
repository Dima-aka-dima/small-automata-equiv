# Equivalence partitioning of the N by N torus for the Game of Life
A partitioning of the states on a small N by N board into equivalence classes based on group action of the $D_4$ x Translations Group ($C_n^2$).

There are $8 n^2$ actions of a group that can equate different states (for $n > 2$). 
If there is an element that commutes with the evolution operator of the system it is called a symmetry and it is reduces the complexity 
of the system.

Here I consider a Game of Life on a torus (periodic square) that is symmetric with respect to $D_4$ group operations and translations.
This reduces the number of states from $2^{n^2}$ to an order of $2^{n^2} / |G|$ where $|G| = 8n^2$ - the size of the group.

To improve performance this code constructs a matrix representation of our group (more specifically of the isomorphic permutation group).

There is also a code that computes the almost exact number of equivalence classes based on Burnside's lemma and some math (much faster than constructing them).

This code main constraint is memmory, not time. Code runs in a minute for a 5 by 5 board, but requires a 100Gb for a 6 by 6.
