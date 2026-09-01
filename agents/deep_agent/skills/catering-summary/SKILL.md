---
name: catering-summary
description: Write an in-flight catering and special-meal summary for a flight. Use this skill when asked to summarize catering, special meals, or provisioning for a flight or route.
---

# Catering Summary Skill

## Format

- **Header**: Flight number and route.
- **Special meals**: One line per IATA special-meal code (e.g., VGML, GFML, KSML) with its quantity; include the total.
- **Dietary / allergen notes**: Any relevant policy or handling notes (cite sources when a policy lookup was used).
- **Loading actions**: What catering must do or confirm, and by when relative to departure.

## Tone

- Concise and operational.
- Always expand an IATA meal code on first use, e.g. "VGML (vegetarian vegan meal)".
- Use exact counts; flag any mismatch between passenger requests and loaded meals.

## Length

- Ideal: under 150 words.
- End with a Sources section only if a policy lookup was used.
