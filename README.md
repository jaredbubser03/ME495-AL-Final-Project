# ME495-AL-Final-Project
Welcome to my final project! This README will walk you through how to use the uploaded code along with general results I found in my own runs.

There are 2 files of note, "diffmpm_final.py" and "diffmpm_main.py". Almost all interactions the user will have are through "diffmpm_main.py".

Creature generation works through defining a genotype, which is a series of tuples representing rectangles, along with a number of
repetitions of that genotype when creating the phenotype. The variable "geno" is of the form [(rect1),(rect2),...,repeats].
Each rectangle should have parameters of x-position, y-position, width, height, actuation #, particle type, and defining corner (which
corner the x- and y-position references).

If it is desired to run the simulation for a given genotype, uncomment lines 21-25 and fill "parent_geno" with the desired genotype.
If it is desired to run the simulation while mutating a given genotype, uncomment lines 27-33 and fill "parent_geno" with the genotype to be mutated.
The final loss and final (post-mutation) genotype are outputted so that the user can save the information.
Mutations can change width and height of different rectangles along with the number of repeats. Mutation rate can be changed in "diffmpm_final.py."

As is, only one simulation can be run at a time, but the commented out lines using subprocess worked at one point in time and could potentially
be used again.

Potential reasons to interact with "diffmpm_final.py" include changing mutation rates or amounts, changing actuation parameters, or changing the
number of iterations run per creature or the visualization of the run. As a default, 80 iterations are run per creature with visualization only
occurring for the final iteration.

For my own tests, I generated 4 generations of 6 children each based on an initial parent, and the results of losses over time are summarized in
"Final Evolution.pdf". It is interesting to note that creatures with 3 repeats performed the best (even though the loss function is the leftmost
point of the creature), and the creatures did improve over time.
