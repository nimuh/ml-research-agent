Dropped into a new vault as `.obsidian/`.

Two settings are load-bearing rather than cosmetic. `useMarkdownLinks: false`
keeps links as `[[wikilinks]]`, which is what Obsidian builds its graph from —
switch it and the vault becomes a folder of pages that merely reference each
other. The `colorGroups` in `graph.json` colour the graph by note `type`, so
sources, claims, methods and experiments are distinguishable at a glance; a
monochrome graph of two hundred nodes tells you nothing.

Obsidian rewrites these files as the user changes settings. That is fine — they
are a starting point, not state. Nothing in the vault depends on them, and
deleting the whole directory loses nothing but the first-open appearance.
