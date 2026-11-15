# AGENTS.md

Guidelines for anyone (human or AI agent) contributing code to AVLite.

## Code principles

- **Clean, minimal OOP.** Each class has one clear responsibility; keep the design as small as possible.
- **Few methods.** Add only the methods a class genuinely needs.
- **Avoid `_private` members.** Use plain names by default; prefix with `_` only when an attribute or method is truly internal. Avoid them as much as possible in API modules, where users are meant to access attributes and methods directly.
- **No reaching into other objects' internals.** Objects interact only through each other's public interface.
- **No redundant helpers.** Don't add wrappers or utilities that duplicate existing functionality.
- **No unnecessary casts.**
- **No comments that restate the code.** Comments explain *why*, not *what*.
