# What this code does

This code reads in jsonl files with a standard format and returns edgelists (a list of edges between node/account pairs) where nodes/accounts are connected if they are coordinated. 

# Coordination metrics
We use the following coordination metrics based on Pacheco et al., (2021).

Pacheco, D., Hui, P.-M., Torres-Lugo, C., Truong, B. T., Flammini, A., & Menczer, F. (2021). Uncovering Coordinated Networks on Social Media: Methods and Case Studies. Proceedings of the International AAAI Conference on Web and Social Media, 15(1), 455-466.


# How to run

python coordination.py <filename>

## Tweet text coordination
We consider accounts coordinated if they use share a long list of hashtags in a given tweet that are in the exact same order. This is highly unlikely to occur by chance.

## Retweet coordination
We back up these results with a complimentary metric, in which accounts are coordinated if they retweet the same content. This is less direct as users may both be superfans of a small set of users, but is nonetheless atypical of most users.

## Tweet time coordination
We finally back up our results with a tweet timing metric, whereby accounts are considered coordinated if they both tweet content at nearly the same times. 

