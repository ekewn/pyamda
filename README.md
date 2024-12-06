# Simple
This library aims to be the simplest, easiest to use (and debug) functional programming library. No complex return types, no weird, base classes. No classes at all actually!

# Approach
We depend greatly on Python's standard library, especially partial, to give the feeling of currying while not sacrificing performance, or generating (hopefully) too much complexity. The general approach taken is to just wrap the imperative code so that usage can be more composable.

# Type Annotation
Type parameters are used where possible, though you're on your own if you're using dictionaries...

# Credit Where Credit is Due
Inspired by the Ramda.js library.

# Contributing
Open to any and all PR's.

## Note
This is a personal project until someone tells me otherwise so I will be iterating quickly. I will try to adhere to semver though so that any API change will be a new minor version at least. 
