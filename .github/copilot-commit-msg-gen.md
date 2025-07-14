Be as concise as possible.
Single-line commit messages are acceptable.
Capture the _reason_ for the change and the _intent_ of the commit.

# Examples

The foolowing diffs show what _not_ to do, and what to do instead.

```diff
- updated README
+ doc: described setup process
```

```diff
- using update_display
+ nb: redrawing the plot with the same display handle
```

```diff
- added SmoothProp class
+ promoted transition logic to its own module
+
+ Experiments in the notebook went well, so copying the SmoothProp to its own
+ module so that it can be reused.
```
