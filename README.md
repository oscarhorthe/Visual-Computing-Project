# Visual_Computing_G10

Mulige ressurser: https://ieeexplore.ieee.org/document/5654319
https://ieeexplore.ieee.org/document/4359370

Beskrivelse av det detect_and_track_w_lamppostmask gjør:
Programmet tracker og gir ID til hver blob ved hjelp av bakground subtraction. Programmet skal til slutt klare å holde styr på hvor lenge hver person er i bildet. Det er et skilt og en lyktestolpe midt i frame som gjør det vanskelig for en foregroundmask å detecte alle i bildet. En lamppostmask har blitt lagt, en en funksjon extendShortBlobsAtSign.m prøver å se forbi skiltet ved å predicte silhouetten bak skiltet. I tillegg har det blitt gjort et forsøk på å lage en logikk der hver gang to mennesker går forbi hverandre og lager en større mask, at programmet holder styr på dette idet de går forbi hverandre og etter de går forbi hverandre. Denne funksjonen fungerer for øyeblikket ikke bra. Det er i tillegg en funksjon som lager en Hue histogram av hver blob, og prøver å bruke denne til å gjenkjenne blober som forsvinner.
