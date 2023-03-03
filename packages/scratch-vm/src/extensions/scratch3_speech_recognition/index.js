/*
 * Resources:
 *  - Text to speech extension written by Scratch Team 2019
 *  - Speech to text extension written by Sayamindu Dasgupta <sayamindu@media.mit.edu>, April 2014
 *  - Knn Classifier model written by Katya3141 https://katya3141.github.io/scratch-gui/teachable-classifier/ August 2019
 */

require("regenerator-runtime/runtime");
const Runtime = require('../../engine/runtime');
const nets = require('nets');
const MathUtil = require('../../util/math-util');

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Clone = require('../../util/clone');
const Cast = require('../../util/cast');
const formatMessage = require('format-message');
const Video = require('../../io/video');
const Timer = require('../../util/timer');
const tf = require('@tensorflow/tfjs');
const knnClassifier = require('@tensorflow-models/knn-classifier');
const use = require('@tensorflow-models/universal-sentence-encoder');
const toxicity = require('@tensorflow-models/toxicity');
var Sentiment = require('sentiment');
//const node = require('@tensorflow/tfjs-node');
//const layers = require('@tensorflow/tfjs-layers');

 
/**
 * Icon svg to be displayed in the blocks category menu, encoded as a data URI.
 * @type {string}
 */
// eslint-disable-next-line max-len
const menuIconURI = 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjxzdmcKICAgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIgogICB4bWxuczpjYz0iaHR0cDovL2NyZWF0aXZlY29tbW9ucy5vcmcvbnMjIgogICB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgeG1sbnM6c29kaXBvZGk9Imh0dHA6Ly9zb2RpcG9kaS5zb3VyY2Vmb3JnZS5uZXQvRFREL3NvZGlwb2RpLTAuZHRkIgogICB4bWxuczppbmtzY2FwZT0iaHR0cDovL3d3dy5pbmtzY2FwZS5vcmcvbmFtZXNwYWNlcy9pbmtzY2FwZSIKICAgaWQ9IkxheWVyXzFfMV8iCiAgIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDY0IDY0IgogICBoZWlnaHQ9IjcyLjc4MDIwNSIKICAgdmlld0JveD0iMCAwIDkuMTk5MTIzNCA5LjA5NzUyNTYiCiAgIHdpZHRoPSI3My41OTI5ODciCiAgIHZlcnNpb249IjEuMSIKICAgaW5rc2NhcGU6dmVyc2lvbj0iMC40OC40IHI5OTM5IgogICBzb2RpcG9kaTpkb2NuYW1lPSJ0ZXh0LWNsYXNzaWZpY2F0aW9uLWJsb2Nrcy1tZW51LnN2ZyI+CiAgPG1ldGFkYXRhCiAgICAgaWQ9Im1ldGFkYXRhMjEiPgogICAgPHJkZjpSREY+CiAgICAgIDxjYzpXb3JrCiAgICAgICAgIHJkZjphYm91dD0iIj4KICAgICAgICA8ZGM6Zm9ybWF0PmltYWdlL3N2Zyt4bWw8L2RjOmZvcm1hdD4KICAgICAgICA8ZGM6dHlwZQogICAgICAgICAgIHJkZjpyZXNvdXJjZT0iaHR0cDovL3B1cmwub3JnL2RjL2RjbWl0eXBlL1N0aWxsSW1hZ2UiIC8+CiAgICAgICAgPGRjOnRpdGxlPjwvZGM6dGl0bGU+CiAgICAgIDwvY2M6V29yaz4KICAgIDwvcmRmOlJERj4KICA8L21ldGFkYXRhPgogIDxkZWZzCiAgICAgaWQ9ImRlZnMxOSIgLz4KICA8c29kaXBvZGk6bmFtZWR2aWV3CiAgICAgcGFnZWNvbG9yPSIjZmZmZmZmIgogICAgIGJvcmRlcmNvbG9yPSIjNjY2NjY2IgogICAgIGJvcmRlcm9wYWNpdHk9IjEiCiAgICAgb2JqZWN0dG9sZXJhbmNlPSIxMCIKICAgICBncmlkdG9sZXJhbmNlPSIxMCIKICAgICBndWlkZXRvbGVyYW5jZT0iMTAiCiAgICAgaW5rc2NhcGU6cGFnZW9wYWNpdHk9IjAiCiAgICAgaW5rc2NhcGU6cGFnZXNoYWRvdz0iMiIKICAgICBpbmtzY2FwZTp3aW5kb3ctd2lkdGg9IjEwMDciCiAgICAgaW5rc2NhcGU6d2luZG93LWhlaWdodD0iNzgzIgogICAgIGlkPSJuYW1lZHZpZXcxNyIKICAgICBzaG93Z3JpZD0iZmFsc2UiCiAgICAgZml0LW1hcmdpbi10b3A9IjAiCiAgICAgZml0LW1hcmdpbi1sZWZ0PSIwIgogICAgIGZpdC1tYXJnaW4tcmlnaHQ9IjAiCiAgICAgZml0LW1hcmdpbi1ib3R0b209IjAiCiAgICAgaW5rc2NhcGU6em9vbT0iMTAuNDI5ODI1IgogICAgIGlua3NjYXBlOmN4PSI0NC45MTIxMTIiCiAgICAgaW5rc2NhcGU6Y3k9IjMzLjEwODQ3MSIKICAgICBpbmtzY2FwZTp3aW5kb3cteD0iNjI3IgogICAgIGlua3NjYXBlOndpbmRvdy15PSIxNzgiCiAgICAgaW5rc2NhcGU6d2luZG93LW1heGltaXplZD0iMCIKICAgICBpbmtzY2FwZTpjdXJyZW50LWxheWVyPSJMYXllcl8xXzFfIiAvPgogIDxwYXRoCiAgICAgc29kaXBvZGk6dHlwZT0iYXJjIgogICAgIHN0eWxlPSJmaWxsOiNhYWZmY2M7ZmlsbC1vcGFjaXR5OjE7c3Ryb2tlOiMwMGFhNDQiCiAgICAgaWQ9InBhdGgyOTkzIgogICAgIHNvZGlwb2RpOmN4PSItNC4zNjI0ODkyIgogICAgIHNvZGlwb2RpOmN5PSIyOC41ODEyMjMiCiAgICAgc29kaXBvZGk6cng9IjMyLjkzNDM5OSIKICAgICBzb2RpcG9kaTpyeT0iMzIuOTM0Mzk5IgogICAgIGQ9Im0gMjguNTcxOTA5LDI4LjU4MTIyMyBhIDMyLjkzNDM5OSwzMi45MzQzOTkgMCAxIDEgLTY1Ljg2ODc5NywwIDMyLjkzNDM5OSwzMi45MzQzOTkgMCAxIDEgNjUuODY4Nzk3LDAgeiIKICAgICB0cmFuc2Zvcm09Im1hdHJpeCgwLjEzNzU1Njk4LDAsMCwwLjEzNjEwMTM3LDUuMjAxMzczMywwLjY1OTI0MTAxKSIgLz4KICA8cGF0aAogICAgIGQ9Im0gMi45NzkxMjMzLDYuMTM3NDI4IGMgMCwwLjEzOTE3NSAtMC4wMzIwMiwwLjI2OTcyOSAtMC4wOTExNCwwLjM4NTUwMyAtMC4xNDA0MDcsMC4yODMyNzcgLTAuNDMzNTM3LDAuNDc2NjQ0IC0wLjc3MTAwNiwwLjQ3NjY0NCAtMC40NzY2NDQsMCAtMC44NjIxNDcsLTAuMzg1NTAzIC0wLjg2MjE0NywtMC44NjIxNDcgMCwtMC40NzY2NDQgMC4zODU1MDMsLTAuODYyMTQ3IDAuODYyMTQ3LC0wLjg2MjE0NyAwLjE3MzY2MSwwIDAuMzMzNzc0LDAuMDUwNSAwLjQ2ODAyMywwLjEzNzk0MyAwLjIzNzcwNiwwLjE1Mzk1NSAwLjM5NDEyNCwwLjQxOTk4OSAwLjM5NDEyNCwwLjcyNDIwNCB6IgogICAgIGlkPSJwYXRoMyIKICAgICBpbmtzY2FwZTpjb25uZWN0b3ItY3VydmF0dXJlPSIwIgogICAgIHN0eWxlPSJmaWxsOiMwMDAwODA7ZmlsbC1vcGFjaXR5OjEiIC8+CiAgPHBhdGgKICAgICBkPSJtIDcuNTAwMjMwNywzLjg1NzIzOTMgYyAwLjQ3NjY0NCwwIDAuODYyMTQ3LDAuMzg1NTAzIDAuODYyMTQ3LDAuODYyMTQ3IDAsMC40NzY2NDQgLTAuMzg1NTAzLDAuODYyMTQ3IC0wLjg2MjE0NywwLjg2MjE0NyAtMC4yMzE1NDgsMCAtMC40NDA5MjYsLTAuMDg5OTEgLTAuNTk0ODgxLC0wLjI0MDE3IC0wLjE2NTA0LC0wLjE1NTE4NiAtMC4yNjcyNjYsLTAuMzc2ODgxIC0wLjI2NzI2NiwtMC42MjE5NzcgMCwtMC4wMzk0MSAwLjAwMjUsLTAuMDc3NTkgMC4wMDg2LC0wLjExNTc3NCAwLjAyNDYzLC0wLjE5MjEzNiAwLjExMzMxMSwtMC4zNjMzMzQgMC4yNDM4NjQsLTAuNDkzODg3IDAuMTU2NDE4LC0wLjE1NjQxOCAwLjM3MDcyMywtMC4yNTI0ODYgMC42MDk2NjEsLTAuMjUyNDg2IHoiCiAgICAgaWQ9InBhdGg3IgogICAgIGlua3NjYXBlOmNvbm5lY3Rvci1jdXJ2YXR1cmU9IjAiCiAgICAgc3R5bGU9ImZpbGw6IzAwMDA4MCIgLz4KICA8cGF0aAogICAgIHN0eWxlPSJmaWxsOiMwMDAwODA7ZmlsbC1vcGFjaXR5OjE7c3Ryb2tlOm5vbmUiCiAgICAgZD0iTSAxLjY4NDIxODQsMy42NDE0NDEgQyAxLjI2Mjk5MDEsMy4zNjI3MjQzIDEuMjAwNjM5NiwyLjc0NjIxMjcgMS41NTU3MTA2LDIuMzcwNzc1NyAxLjkwMjI1OTEsMi4wMDQzNTAxIDIuNDk1MTc4NSwyLjA2MzM0NTYgMi43Nzk0OTE4LDIuNDkyNTQxOSAzLjI0NDkyNjgsMy4xOTUxNTc3IDIuMzgwMzU2Myw0LjEwMjA1ODMgMS42ODQyMTg0LDMuNjQxNDQxIHoiCiAgICAgaWQ9InBhdGgyOTg4IgogICAgIGlua3NjYXBlOmNvbm5lY3Rvci1jdXJ2YXR1cmU9IjAiIC8+CiAgPHBhdGgKICAgICBzdHlsZT0iZmlsbDojMDAwMDgwO2ZpbGwtb3BhY2l0eToxO3N0cm9rZTpub25lIgogICAgIGQ9Ik0gNC4zMDQ5MDQ0LDIuNTc3NjY4MiBDIDQuMDEyNzcwNiwyLjMyODI3MTUgMy45MzQyNTU5LDIuMDAxMTIzMSA0LjA4MTAxMTIsMS42NDQ3Njk3IDQuMzA2NDE4NCwxLjA5NzQzMjkgNC45Nzg0MTEzLDAuOTI2NTkzODcgNS4zODg2NTEzLDEuMzEyMzMxNiA2LjE5NzgyNzQsMi4wNzMxNzg3IDUuMTQ4NzM2NSwzLjI5ODA1MjUgNC4zMDQ5MDQ0LDIuNTc3NjY4MiB6IgogICAgIGlkPSJwYXRoMjk5MCIKICAgICBpbmtzY2FwZTpjb25uZWN0b3ItY3VydmF0dXJlPSIwIiAvPgogIDxwYXRoCiAgICAgc3R5bGU9ImZpbGw6IzAwMDA4MDtmaWxsLW9wYWNpdHk6MTtzdHJva2U6bm9uZSIKICAgICBkPSJNIDQuMzc4NDk3Miw0Ljk3NTcwMTEgQyAzLjgyNzgzNjUsNC42NDA5NTU0IDMuODkyODQ1NiwzLjg1MDQyNzMgNC40OTM4MTgyLDMuNTczMzY5NCA0Ljg2NjA4OCwzLjQwMTc0NjkgNS4yODkzODIyLDMuNTMwMjgyNCA1LjUxMDExNzUsMy44ODE5NzM4IDUuNzMwMTg2Miw0LjIzMjYwMzMgNS43MDg5NzY2LDQuNTI0NDQ5MiA1LjQ0MjU2MDksNC44MTE1NjA4IDUuMTQ1NjQ4OCw1LjEzMTUzNzcgNC43MzgyMzQzLDUuMTk0Mzg0NiA0LjM3ODQ5NzIsNC45NzU3MDExIHoiCiAgICAgaWQ9InBhdGgyOTkyIgogICAgIGlua3NjYXBlOmNvbm5lY3Rvci1jdXJ2YXR1cmU9IjAiIC8+CiAgPHBhdGgKICAgICBzdHlsZT0iZmlsbDojMDAwMDgwO2ZpbGwtb3BhY2l0eToxO3N0cm9rZTpub25lIgogICAgIGQ9Ik0gNC4zMjQ2MjY4LDcuOTgyMDc1MSBDIDQuMDM2Njg0OCw3LjczNzk3NTQgMy45NTkyOTY1LDcuNDE3Nzc1MiA0LjEwMzk0NjIsNy4wNjg5OTA0IDQuMzI2MTE5LDYuNTMzMjc4MiA0Ljk4ODQ2OTUsNi4zNjYwNjc2IDUuMzkyODIzLDYuNzQzNjEyOCA2LjE5MDM4ODUsNy40ODgzMDA1IDUuMTU2MzUwNyw4LjY4NzE1OTcgNC4zMjQ2MjY4LDcuOTgyMDc1MSB6IgogICAgIGlkPSJwYXRoMjk5NCIKICAgICBpbmtzY2FwZTpjb25uZWN0b3ItY3VydmF0dXJlPSIwIiAvPgogIDxwYXRoCiAgICAgc3R5bGU9ImZpbGw6I2Y2ZGI0MDtmaWxsLW9wYWNpdHk6MTtzdHJva2U6bm9uZSIKICAgICBkPSIiCiAgICAgaWQ9InBhdGgzMDAxIgogICAgIGlua3NjYXBlOmNvbm5lY3Rvci1jdXJ2YXR1cmU9IjAiIC8+CiAgPHBhdGgKICAgICBkPSJtIDIuNTQzNjE2Myw1LjI1MDY0OCBjIC0wLjEyOTMyMiwtMC4wNjI0NCAtMC4yNzM2NywtMC4wOTg1MyAtMC40MjY2NCwtMC4wOTg1MyAtMC41NDMyNzUsMCAtMC45ODUzMTEsMC40NDIwMzUgLTAuOTg1MzExLDAuOTg1MzExIDAsMC41NDMyNzYgMC40NDIwMzYsMC45ODUzMTEgMC45ODUzMTEsMC45ODUzMTEgMC4zNDA5MTgsMCAwLjY0MTgwNywtMC4xNzQxNTQgMC44MTg3OTMsLTAuNDM4MDk0IGwgMC45MzA3NSwwLjQ2NTkyOSBjIC0wLjAxNjAxLDAuMDcwMzMgLTAuMDI1MjUsMC4xNDMzNjIgLTAuMDI1MjUsMC4yMTg0OTIgMCwwLjU0MzI3NiAwLjQ0MjAzNSwwLjk4NTMxMSAwLjk4NTMxMSwwLjk4NTMxMSAwLjU0MzI3NSwwIDAuOTg1MzEsLTAuNDQyMDM1IDAuOTg1MzEsLTAuOTg1MzExIDAsLTAuMjA0MjA1IC0wLjA2MjQ0LC0wLjM5NDAwMSAtMC4xNjkzNSwtMC41NTE1MjcgbCAxLjMwNzg3NywtMS4zNjc0ODkgYyAwLjE2NDA1NCwwLjEyMTY4NiAwLjM2NjI4OSwwLjE5NDcyMiAwLjU4NTc2NywwLjE5NDcyMiAwLjU0MzI3NiwwIDAuOTg1MzExLC0wLjQ0MjAzNSAwLjk4NTMxMSwtMC45ODUzMSAwLC0wLjU0MzI3NiAtMC40NDIwMzUsLTAuOTg1MzExIC0wLjk4NTMxMSwtMC45ODUzMTEgLTAuMjI3MzYsMCAtMC40MzYyNDYsMC4wNzgwOSAtMC42MDMyNTYsMC4yMDc5MDEgTCA1LjYwMzk5MDMsMi41NTMxMTUgYyAwLjEyOTgxNSwtMC4xNjcwMSAwLjIwNzksLTAuMzc1ODk2IDAuMjA3OSwtMC42MDMyNTcgMCwtMC41NDMyNzUgLTAuNDQyMDM1LC0wLjk4NTMxMTAyIC0wLjk4NTMxLC0wLjk4NTMxMTAyIC0wLjU0MzI3NiwwIC0wLjk4NTMxMSwwLjQ0MjAzNjAyIC0wLjk4NTMxMSwwLjk4NTMxMTAyIDAsMC4wNzQ4OCAwLjAwOTEsMC4xNDc2NzQgMC4wMjUxMiwwLjIxNzg3NyBMIDIuOTkyOTExMywyLjQ4NTk5IEMgMi44MjkyMzMzLDIuMTY4MTA0IDIuNDk4NDE1MywxLjk0OTg1NyAyLjExNjk3NjMsMS45NDk4NTcgYyAtMC41NDMyNzUsMCAtMC45ODUzMTEsMC40NDIwMzUgLTAuOTg1MzExLDAuOTg1MzExIDAsMC41NDMyNzYgMC40NDIwMzYsMC45ODUzMTEgMC45ODUzMTEsMC45ODUzMTEgMC4xNDAxNjEsMCAwLjI3MzMwMSwtMC4wMjk4MSAwLjM5NDEyNSwtMC4wODI4OSBsIDAuNDUzMzY2LDAuNzYzIHogbSAtMC40MjY2NCwxLjYyNTc2MyBjIC0wLjQwNzU0OSwwIC0wLjczODk4MywtMC4zMzE0MzQgLTAuNzM4OTgzLC0wLjczODk4MyAwLC0wLjQwNzU0OSAwLjMzMTQzNCwtMC43Mzg5ODMgMC43Mzg5ODMsLTAuNzM4OTgzIDAuNDA3NTQ5LDAgMC43Mzg5ODMsMC4zMzE0MzQgMC43Mzg5ODMsMC43Mzg5ODMgMCwwLjQwNzU0OSAtMC4zMzE0MzQsMC43Mzg5ODMgLTAuNzM4OTgzLDAuNzM4OTgzIHogbSAwLjkyODUzMiwtMC40MTI0NzYgYyAwLjAzNjA5LC0wLjEwMjM0OSAwLjA1Njc4LC0wLjIxMTk2NSAwLjA1Njc4LC0wLjMyNjUwNyAwLC0wLjMwMjEyMSAtMC4xMzY5NTgsLTAuNTcyNTg5IC0wLjM1MTYzMywtMC43NTMzOTMgbCAwLjM1NDIxOSwtMC41NDcwOTQgMS4wNjc5NTQsMS43OTcwODMgYyAtMC4wODk5MSwwLjA4MDA2IC0wLjE2NTE2MywwLjE3NTg3OCAtMC4yMjA5NTYsMC4yODM2NDcgeiBtIDEuNzgxMDczLC0zLjUyODc2NyBjIDAuMjI3MzYsMCAwLjQzNjI0NiwtMC4wNzgwOSAwLjYwMzI1NiwtMC4yMDc5MDEgbCAxLjMyOTA2MSwxLjMyOTA2MSBjIC0wLjA3OTY5LDAuMTAyNDczIC0wLjEzOTI5OCwwLjIyMDgzMyAtMC4xNzM2NjEsMC4zNDk2NjMgTCA1LjgxMTUyMjMsNC4yOTk1NzcgYyAtMS4yM2UtNCwtMC4wMDMyIDMuNjllLTQsLTAuMDA2NCAzLjY5ZS00LC0wLjAwOTYgMCwtMC41NDMyNzUgLTAuNDQyMDM1LC0wLjk4NTMxMSAtMC45ODUzMSwtMC45ODUzMTEgLTAuMzQwOTE4LDAgLTAuNjQxOTMsMC4xNzQxNTQgLTAuODE4NzkzLDAuNDM4MDk0IEwgMy44NjA4NTMzLDMuNjY5MzUgNC40MDAwNjQzLDIuODM2NjM5IGMgMC4xMjkxOTksMC4wNjI0NCAwLjI3MzU0NywwLjA5ODUzIDAuNDI2NTE3LDAuMDk4NTMgeiBtIDAuMTIzMTY0LDMuNDU3MDg2IFYgNS4yNjY2NTkgYyAwLjQwMDI4MiwtMC4wNTAyNSAwLjcyNTkyNywtMC4zNDExNjMgMC44Mjc5MDcsLTAuNzIzMjE4IGwgMC43NzM3MTUsMC4xMDY0MTQgYyAwLDAuMDAzMiAtNC45MmUtNCwwLjAwNjQgLTQuOTJlLTQsMC4wMDk2IDAsMC4yMzUyNDMgMC4wODMwMSwwLjQ1MTE0OSAwLjIyMTA3OSwwLjYyMDc0NiBMIDUuNDc4NzMzMyw2LjYzMjQxNiBDIDUuMzMzNzY5Myw2LjUwMzk1NiA1LjE1MTM2NDMsNi40MTc2MTkgNC45NDk3NDUzLDYuMzkyMjQ3IHogTSA0LjgyNjU4MTMsMy41NTA5ODcgYyAwLjQwNzU0OSwwIDAuNzM4OTgzLDAuMzMxNDM0IDAuNzM4OTgzLDAuNzM4OTgzIDAsMC40MDc1NDkgLTAuMzMxNDM0LDAuNzM4OTgzIC0wLjczODk4MywwLjczODk4MyAtMC40MDc1NDksMCAtMC43Mzg5ODMsLTAuMzMxNDM0IC0wLjczODk4MywtMC43Mzg5ODMgMCwtMC40MDc1NDkgMC4zMzE0MzQsLTAuNzM4OTgzIDAuNzM4OTgzLC0wLjczODk4MyB6IG0gLTAuOTI4NTMyLDAuNDEyNDc2IGMgLTAuMDM2MDksMC4xMDIzNDkgLTAuMDU2NzgsMC4yMTE5NjUgLTAuMDU2NzgsMC4zMjY1MDcgMCwwLjUwMTUyMyAwLjM3Njg4MSwwLjkxNTg0NyAwLjg2MjE0NywwLjk3NjgxMyB2IDEuMTI1NTk0IGMgLTAuMTE2NzU5LDAuMDE0NjYgLTAuMjI2OTkxLDAuMDQ5NjQgLTAuMzI3MzcsMC4xMDE0ODcgTCAzLjI1NDI3MDMsNC42MDYxMzIgMy43MjYxMTEzLDMuODc3NDk1IHogbSAwLjkyODUzMiw0LjE0NDU4NiBjIC0wLjQwNzU0OSwwIC0wLjczODk4MywtMC4zMzE0MzQgLTAuNzM4OTgzLC0wLjczODk4MyAwLC0wLjQwNzU0OSAwLjMzMTQzNCwtMC43Mzg5ODMgMC43Mzg5ODMsLTAuNzM4OTgzIDAuNDA3NTQ5LDAgMC43Mzg5ODMsMC4zMzE0MzQgMC43Mzg5ODMsMC43Mzg5ODMgMCwwLjQwNzU0OSAtMC4zMzE0MzQsMC43Mzg5ODMgLTAuNzM4OTgzLDAuNzM4OTgzIHogbSAyLjcwOTYwNCwtNC4xODc1NyBjIDAuNDA3NTQ5LDAgMC43Mzg5ODMsMC4zMzE0MzQgMC43Mzg5ODMsMC43Mzg5ODMgMCwwLjQwNzU0OSAtMC4zMzE0MzQsMC43Mzg5ODMgLTAuNzM4OTgzLDAuNzM4OTgzIC0wLjQwNzU0OSwwIC0wLjczODk4MywtMC4zMzE0MzQgLTAuNzM4OTgzLC0wLjczODk4MyAwLC0wLjQwNzU0OSAwLjMzMTQzNCwtMC43Mzg5ODMgMC43Mzg5ODMsLTAuNzM4OTgzIHogTSA0LjgyNjU4MTMsMS4yMTA4NzQgYyAwLjQwNzU0OSwwIDAuNzM4OTgzLDAuMzMxNDM0IDAuNzM4OTgzLDAuNzM4OTgzIDAsMC40MDc1NDkgLTAuMzMxNDM0LDAuNzM4OTgzIC0wLjczODk4MywwLjczODk4MyAtMC40MDc1NDksMCAtMC43Mzg5ODMsLTAuMzMxNDM0IC0wLjczODk4MywtMC43Mzg5ODMgMCwtMC40MDc1NDkgMC4zMzE0MzQsLTAuNzM4OTgzIDAuNzM4OTgzLC0wLjczODk4MyB6IG0gLTAuODc1ODE4LDEuMTg4MTYyIGMgMC4wNjAyMywwLjExNjg4MiAwLjE0MjYyMywwLjIyMDIxNyAwLjI0MjI2MywwLjMwNDIxNCBMIDMuNjM5MjgxMywzLjU1ODUgMy4wNDU2MzIzLDMuMjYxNjc1IGMgMC4wMzU5NiwtMC4xMDIzNDkgMC4wNTY2NSwtMC4yMTE5NjUgMC4wNTY2NSwtMC4zMjY1MDcgMCwtMC4wNzQ4OCAtMC4wMDkxLC0wLjE0NzY3MyAtMC4wMjUxMiwtMC4yMTc4NzcgeiBtIC0yLjU3Mjc3LDAuNTM2MTMyIGMgMCwtMC40MDc1NDkgMC4zMzE0MzQsLTAuNzM4OTgzIDAuNzM4OTgzLC0wLjczODk4MyAwLjQwNzU0OSwwIDAuNzM4OTgzLDAuMzMxNDM0IDAuNzM4OTgzLDAuNzM4OTgzIDAsMC40MDc1NDkgLTAuMzMxNDM0LDAuNzM4OTgzIC0wLjczODk4MywwLjczODk4MyAtMC40MDc1NDksMCAtMC43Mzg5ODMsLTAuMzMxNDM0IC0wLjczODk4MywtMC43Mzg5ODMgeiBtIDEuMzQ0MzM0LDAuNzc1ODA5IGMgMC4wODI1MiwtMC4wNjQ1NCAwLjE1NDk0LC0wLjE0MTM5MiAwLjIxMzQ0MiwtMC4yMjg1OTIgbCAwLjU2ODY0OCwwLjI4NDI2MiAtMC4zOTA1NTMsMC42MDMxMzMgeiIKICAgICBpZD0icGF0aDE1IgogICAgIGlua3NjYXBlOmNvbm5lY3Rvci1jdXJ2YXR1cmU9IjAiCiAgICAgc3R5bGU9ImZpbGw6IzAwMDAwMCIgLz4KPC9zdmc+Cg==';

/**
 * Icon svg to be displayed at the left edge of each extension block, encoded as a data URI.
 * @type {string}
 */
// eslint-disable-next-line max-len
const blockIconURI = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlcAAAJXCAMAAAH+4rEgAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALMw9IgAAAD/dFJOUwABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4fICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj9AQUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVpbXF1eX2BhYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ent8fX5/gIGCg4SFhoeIiYqLjI2Oj5CRkpOUlZaXmJmam5ydnp+goaKjpKWmp6ipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsPExcbHyMnKy8zNzs/Q0dLT1NXW19jZ2tvc3d7f4OHi4+Tl5ufo6err7O3u7/Dx8vP09fb3+Pn6+/z9/usI2TUAAAAJcEhZcwAAFxEAABcRAcom8z8AACq+SURBVHhe7d15wB3T/fjxJ/sukpBEYkutpbaEoJYiitqK0igt0lZtX3upFm1t1RYtv6JF26C21q6lomrXil1qJyEJQYQkEtmT53fvmfe9z8ydmXPm3DvrvZ/XH/LMZz7nnM+c53ie+9w7S5uI3XntJZuz0YhyP3idUJ3opeJDwnVYny5c2GPtf7T3YJ+la2ldg712aOvDbhu0DEBCdLNoGICM6GgXiJSoxtAsEDlR0SrY2iRFRKsQJEVEoxAkRUSjECRFsxKNQpAVzUAahSArGukral8bjBw5KkpfI0eOdBpoPKNSzX05/9WTvlDKc/6rJ32hlOf8V0/6QinP+a+e9IVSnvNfPekLpTznv3rx99VbNQlVynP+q+f0VU7VcBJUvs7j1VQNJ0Hl65xeTdVwElS+VjU1XNdqmgFJqlEIV5peR26YFaXd/Ur/ltP0Sklblf7ZWzULRNaC8r968zqyg/2DveV/TCppqmGA8r5N+Nekkh/SWWXXYvWFSaVBZ9W2htrztcoXRtUmAZVV4xFmXimlznW+Wqg6qHKCrsHMFpRyd+Tr+8sNHUSsunKyd+Prkq3OOqYXX5aUd37K11GU8+fxdY3yLouySkJbhO7QUG18jYKjRuVvQMkQNkt4I2VlNq04TWux01YXmruwpy7T6EM5g2ADnI7YECJeq5dX101sNOJqtU4dhOp0OL1UEK4HPbjcyx5rtPdinyUa12KvFZr6HMB+C+fT1I8ECzQMQkZkb9EuCCmR0SzQInKiolkwciJ6ilbBSIqIRiFIiohGIX5GVjQ0CjGJrGhoFGIKWdHQKIT01cF5b5hGIcp9RXkPuZTq/Fej3FfpH5Wv4+SoJqGkrw5OjmoSSvrq4OSoJqGkrw5OjmoSSvrq4OSoJqGkrw5OjmoSKs6+Jlbz9Jwc1STUUdU8PSdHNQnVkafn5Jyh2oQpJezm/KNHjmoTprT/JecfvVLOLs4/oZ539l+t8nU+L2WV/tlDtQpWTuMfg47kMK4sA7I6qWZBynsf4F+DSnr53yCVpKPVFwau/AA/LO/qWfpC5ZiUW1S/qHVudY/6wqiU2LnyRQ0VbltS/cqoo1Ftbx3Rq/jSyNXM1dsMIp7dZjXZR5y1K1+V2XXVdk54fnfLrtra7iq3uJwNt3Lcrqu2ts0DG/0jMGqmmrUvZavMidTRVc3btFXD2GuL5i4/ZU9d6MNBrAFOP5+xJYTIqa7Tnf9ZO8wfyK6sPEohQUhJHcNrrERmeq5iZBPS0/E0g0ZBk+StxYARfUSzhDGaBRomiqGsbEHb5DCQpXG0TgrDWOtO+2QwSB3oIBHhf5EaLaSLJDBEXegiCYxQl23oI36Bp2xFdQ+dxM9weqde+Z27ZEhZNqQsGzkqq3wqc4n6Oo6ynA3n6wZIWTakLBtSlg0py4aUZUPKsiFl2ZCybEhZNqQsGwUoa4DzdX3eUl0kURZ91ud6dxfO1w2IrSyu6XI2nK8bcIu7n/IVmPVyehjk3mhA+dLMjn6cjXr82engMmfL2WiEp5/XnK060IF3qwFOP+96tuzRnPZrstWA7zo9sVVnXc7ZSW1tHzmbbDXE6WkZW3XVRdNK493Zakh/p6+D2VTnqdmhYfWQ2GrQi05ng9m0nbD1aBVzVW1ty5zu1mLTeHMCN34wlBFhKwZ0eBebZYQMRpNdRoitWHwS0Oc3iIWaSqKyH0E2YzKaXn/LNlYQ9ludDBCNuaoS+g245d+F7MFT/Yl3YE/7qWzHyTnPr+QXBKKiWQJT5fgr3dsMcActEiuq7FyGKPF/q/zeJ7eESGIYRtmXWJAZ5JQ9SSxRWzJYxSTvaQ4XLSJeQTwNjGh0PvkpepChwyR/rojGPsuposOTHb/VhRBCCJEDOz3BL+kZR/QmlLUDqajD39mTISqptQ+7MzGYIoJ8n5z0UUCYbFbZKYweLqUz0j0YWo/c9DCuScov6hnVzH3VZeIYM4oxNEkBI0azCo0Sx3hR0Spp7zBcZLRLFp9xWZhGy0Qxlg1aJomPPe3QNkEMZGc/GifmXQayROvEMIytbWmekBsZxhrtE8Ig9mifjPovXbmdHhKh7n5QH3pIBEPUgx4SwRD1SPIFIUPU40W6SMDBDFEX+kjAvYxQF/pIAB+l14c+EsAA9aGPBDBAfegjAQxQH/pIAAPUhz4SwAD1oY8EMEB96CMBDFAf+kgAA9SHPhLAAPWhjwQwQH3oIwEMUB/6SAAD1Ic+EsAA9aGPmIxUnK8ZoD5OF6NUdxs4Gw1w9+l8XSd3F087Gw1w+nF/XSd3F1KWibsLKcvE3YWUZeLuQsoycXchZZm4u5CyTNxdSFkm7i6kLBN3F1KWibsLKcvE3YWUZeLuQsoycXchZZm4u5CyTNxdSFkm7i6kLBN3F1KWibsLKcvE3YWUZeLuQsoycXchZZm4u5CyTNxdtERZrkvw7TldOF/HW9blzkZ9nC6cr+MtazNnoy5znS6cjcecjQY4/fR1b9TlCncPjT8x3ennK+6Nujgd0EPjN/dw+vmVe6MuTgf0sDFb9XP6ec7Z2NPZqofTAd2x0QCnH2+vdRjr6YCNBlzj6cjZqAPtd/dsNYDnvrC1mrNlbTntP3A22WqE01HlYkNnyxqtaT6PrUY4PVUeMHi8s2mL1nR2BFuNWOB0xVZ900Xbtu29m43ghwJbbV90Nq3MoW3lmNhqjNNV9apkZ9MKLStt+fXYIKev2r4tVH+mH+Fsx/PAreOczqrnXK3nbEdWfRyn7wAbU9tbx42lIqFV9YfeC2w2qvws1JKOBxjxP2c0tCnxBRrk628xkQhoUbKOE6jeF7FhTn/uq34jzxf5Zf5Io/w93knIgOyy+/yhRi10elzEZtnGTkhrArllleu82IwHfe7JpkIsXCcSFWIfsxmPP9Mrm44NCQb7I1mOz4iyGRd6ren2WKJ+D5GBKwhzXm186Lf2cPv5bwpWciR7K3YmvoDt+PyInv3fhhE1Py32It6heiNEtuM0l66D+x66z4ln/Wjshmx5VR+Lzna86Nu694tp1r4JgZjRe3t7VwKRzKdRYw9w1qH/9vbbCERAi/b2GwkkgBFKCJj8nvSEr8lljJLpRHRWJrfkbEIJYZQy0z2H3A9bTGi1d5jIQGXvEwvydXIUYkny3qZiPFGvoUvZrXxONGGMVrHoIuKOkZOIV/hvvpuQNRkwindok4pfMqgR+alx3VU3XOX+7Wkaydhh7icvfT+hAj/PjbYzMPi/FNJh+d7sy5jnr1liQgghhBBCCCGEvVFXvsmfl37LHjihC2mtbufXmROjC4fTpCX145F5FhYdRdvWchuHb29BTt6cSsmqnH9Xv5/TU7PrMpkDbtC36K+JHcqhxmFmc/+SnMBhxmYrOm4+9T8XWcN/8kczqDzhJ3YbMUDzGMeRJWFxDwZpDt3CH50bizsZpxn8mGNKUD+GKrzqyVpJOoHBis3iUeQNafwywuwN41iSt4QRi4sreFJR9NnaiONIR/VqpkLqwlGkZSbjFpLnrMk03MrABfQvDiFFWzN04WzBAaSKsQuH8tP1MwYvmDjf57PA6AUzi+pT9hOGL5QvUHza4r2SMCWRz+OPWx8KKJK3qD1136GAIqH09HkvqC2EXpSevv9QQYG4Lu9L2RtUUCADKT19U6igQGSyLMhkWZDJsiCTZUEmy4JMlgWZLAsyWRZksizIZAV4lhIVYmU5mSxiyjPEsvMMlSjEynI4WdmfOiKTZUEmy4JMlgWZLAsyWRZksizIZFmQybIgk2VBJsuCTJYFmSwLMlkWZLIsyGRZkMmyIJNlQSbLgkyWBZksCzJZFmSyLMhkWZDJsiCTZUEmy4JMlgWZLAsyWRZksizIZFmQybIgk2VBJsuCTJYFmSwLMlkWZLIsyGRZkMmyIJNlQSbLgkyWhbDJ6k8ofS9TgUJMye9kdSWUPs9DfYkp+Z0sT5mp+gUFKMSUHE/Ww8RStwsFKMSU7CfrKSpRiClHEksd4zuIKf8llp1/UolCzEEsbZ6bHHluwXsfwexcRiXKugSVVwmmzPNIi40JKpcSzM6BVKJ8j6DSl2DKGN1xLEHlAILZ6Uwlyj0EHS8RTdWXGNxxH1GlE8EMUYnyKTFHWs8YcHuSsfEZYYVYlmZQirIaQccPiKZnOSNjDcLKewSztBe1KBcQxPOEU7M6A+NiwsrXCGbJ+3cNwYqUbwj/bYZFJ8KOrkQz5Xms49EEKxYQT8VvGLTiZOJKPu6rP4BqlJofGm1tS9iRAt/rKM9zpPoTzJjndjR3EKxK5RFFZYcyYNXf2aFk/4chqMexGcGq+9mRsPUYrmoUOxwEs/c9CnIQ7DCGHUlaylgu7HGMI5gD71CSMptgh07L2ZWYPzCSy1x2KZMJ5gI1OQLuBJ3w+zWM4uZ5ny0//xOWeV4qtz9I1C2mZ/wGOYgh3B5ln2MY0Zw4gLIcLxF165zQi4jf0r+H9+2h/YjmxoUU5phK1COJW8T/jr69PH+vtp9PNEd+R2kgWsPzm6BxIU/CZy8Cl17WTqU4+F5vOU5id+MW9KTLGt7XV+0nEc6ZQygPns/v3DyfcdRrfzrz+TcJ+Cbh3BlEgRWrEPd7gIw6+f6yqRpMRsVA4nlU8xvvWcJBvvE5SZYmdaeDIC+QhJw/S/Meyqw4jHiw7Szfwbk9fK2WfZe0ituI59aaFFq1DTtC7XYvmTqfnLMq6aG+TGrVGuzIM+8fGiUhvxdrDD7l8am+162zX75uD/Yb+B6mmP3nz5H4X3wm/kSqIxioQ3Geh1/zIqLkUfYk4gkG6ZDbFwyBxlO1yxh2xWw3une5hl3FcTeVuywazr7YrL6Irl1uZ1+xXEX1biti/PRuL88HEricncVzOEfgdXc/djegv+fTiKqaDw4LpnvIy84XB5FQh1VeoZMa87uRUGAncix+96xJSmRrBS+osmNJKTzPh9a1Hjp0CGkaQ779EOmB/kpakwj45Vhr6UuXH7LZGoMH9Ovds2fvfgMGr7HZoZdPinC2RDF//Rnsw8HFai86b0aXcoyxuIROm9mRn3CwDZiVo4+YE9fj8LpPRpp/qO7tv+bVeazVWSP/DPo4tfWscvyEoL9dlOUT/i/P76QLIYQQQgghhBBCCCGEEEIIIUQqem522A3T+Ry14r+/2LkIl0Wk69j7w68Bfvvq0WSJ/rfOY1bCzTg0FzfZydbgc5gNo3d2pUmL+urLTEQ019KsBR1qfzbW6wNo21q+zeFberbmNmstYKc5HLu9e+miRfRq7D4OJ9JNK/DcRa4es1vlFWqPaRxxI86js+a2B0fboBlNcOmSyY0ca+NG0mPTiuP/v4qc3kgmJn2XcZjxuJNum9G6HGNsnqPj5rM5RxijaXTdbEZzfLGaSefNZROOLma5umVmTIZwbLHzPL2jKfThyBLwEEM0i06zObAknMMgTeI1DisZX2aUpvBbDiopvRmnCWzJISXmMwYqvi7x/mUTZDxDFd5zHFCSot0AL/c8D5BJDIMVXOjV37HyPQ+jiG7gYJJmfROk/FmFQ0nc+wxYYDM5lOTtwIiFtSkHkgaGLKwPOI40eJ4sWTzbchip+JxBCyrBB5wEyP6hoA1I7ZegYw7DFtJEDiItRT7fiENITdCTewriDA4hPQxcQN5n/qThZEYunMQ+ugn3DkMXzpkcQJpy8khQa6k+gRYFva1mT8pP1ZsMXjB7U366ujB6sWTypP+8PbkxosVUn65i3jCZ4lPmf/xtAexJ8SnzPeG8CP5I8Wkr4rncaXyCGmRPxi8S88WmyfgN4xcJpafu34xfILVP6kzN2xRQIAmcoB3NQgookN0pPX0UUCDjqDx9xXvRcDaVp0//aNo8uoLK0zeCCoojvusEbW1EBcVxB5Wnr3inQ9Y+Lzs9xbteNfzJgUmTuYpO5io6mavoZK6ik7mKTuYqOpmr6GSuopO5ik7mKjqZq+hkrqKTuYpO5io6mavoZK6ik7mKTuYqOpmr6GSuosvvXI0a2WF9YmX5mKsNqKxsFLHsUKDyBLGyfMzVM8QUYtmhDuVxYmX5mKuniSnEskMdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRhyJzpUcdisyVHnUoMld61KHIXOlRh+KeqzuJpW9zKigrxlz9lVj6NqaCsmLM1VXE0rcOFZQVY67OJ5a+oVRQVoy5OoZY+npTQVkx5mpvYumjAKUYc7U1sdQtpwClGHO1BrHUTacApRhz1YlY6p6kACW/c+UpcxnBtF3L+MpzBBVi2aEO5VliyqsE03YI4yuTCCrEskMdyv+IKbcSTNsAxldeIagQyw51KG8QU44gmLIVDO94k6hCLDsrKKTsXWJKL4Ipm8/wjmlEy7yzmIWlVFL2ITFHNj/cb2N0x8dEy5YSy84CKinzfk8/Ipou97tX3uoWEMvOZ1RS5l3lxxNNVw9GdxBU5hLLzkwqUToRVPoRTNU8Bnd4XhB/RDA7D1CJ8gWCDoKpOpuxHRsSVe4nmJ2zqUTZn6Aji7f71mZsx1iiypkEs7MDlSjnEnRsSjRFNT+TfklY2Y5gdvpSiTKBINy/hdJxFSPjYcJKX4LZ6U4lSs2v5fGE0+P9Ldi2hLDSjWCGqMTRlaCjK9HU1Pwv6Pk+Zv8nTlvb65SirEUQnhcUKfgK42JtwsprBLN0IbUoRxLEZoTT4nl919Z2AmHlAoJZ2pValLcJVnh+YCTuDEateI+4sgvBLK1ELQ6CFYcTTsWKzoxaQdzRj2CmFlKM0odgxSLiabiaMSsGEFey/8u57DGqUS4mWHEU8TQwZNUVxJVHCGbL8wmz5xOnMvd7gck6jRGrZrBDOYpgtrpQjaM70YpRxJPHgFW9iTu6EM2Y55ddzauGmre8E7QX41Udxw5lCcGs/Y16lPcIVq3CjoS9yXAdPC+EbyKYtdWpx7ES0arfsCNZvr/2VmaHYzWimaMex3kEO8xnT5L8P7ovZo+DYPY8/xP6yxrEjgS9zFAu7HHcTDB7o6nIsS3RDqewJzne9zfKdmKPY0uiObCYkpS3CLo8z66kbMM4LlPYpSwimAfXU5NjDaIu7EnILxjFZQS7HJ5zZzLmfTka8IHJQHYlwnMqEx5inyMnL0Qdn1CUI+BP+u3YlYDZDOHWn32OWUTzYUuqcjxF1O1g9sWPATy8PyC3IJoTyynLUfvOTNnJ7IvZcvcp2hXeZeU52zYHvC8LJhH1SObqgEH07uE9p/AEonnR2fveyyjCHmeyM0796dtjG3Y6lte8C5897wfyi4l6Hc3e2Cyp+TwQ3h8IVxDNESrDqUS99mdvTELOfDmL3SCaJxdQGoI/5d2AvbF4mE5r9GA3fko4V6gNIR9ddpnN/sZdSJe1vG8uZn+WaBDP+5ABb5DC+6ZE/UbTX62aMo4mnDM1SybohU/ZLuxvyOthZ3J4ztsJflGfB5tTH7xn2rq9QUb9vkdPfjXnMX2JcO54ruhob7+DsN/XyajT47UfMHeouXz/BcL505MKK7ynRHo08Cb8UveF8jU8Jz2WBL/+ygXPiSklmk8EunguV4tu0RF0EGQ4SRXHEs+lDyiyQvd9XfWfJNk4mcaBai9r8X3+liveM+l0P9/LBlxHWkSLx9AwxOfkVeTgrEed2rep3ice5lDPOVJaD65MmzAfkljxDeK5NZFCK14hHmrls2uPMciEr5EeznMyZsl/iOdXJ89nOiWei1WDrbSH57ykWosu8l6PEaz2s6JFuXsrxs93/kLE1zi7jn/5U1pULZ32xOkRP12veW1Xc31qXh1CsVX+0zJC9dxgzNijf3TWWWedOG7f0TbnILzNWFVj2ZFzN1Bu1afsSI7v3Yvr2JF7tT9l29sHsicZ/tOWXmVPAfjfpAr4FD022zNGh0/YUwTez6EV39mcsfkxI3RYkavPmU36ULXLi+yKWSffL8D29p7sK4iAc66WJ/Fe0mYB5zmbXt/nTu2f/GV/YV98fL9yS3JzumN03rNIHTEvrU3o1sN9C7rC8FziUXEvO+MQ+J5O4AfR+dct8Cov7TtQFk6jP49F/pMhi2Iqh+C1H3sbEfwJ9lT2FtIEDsJr1oHsrtdB3nPjKu5jd0FdxGHUWPKd8I9iTDqPC7mCMw9XnjZkRw7E58ohZNgZGnqHjC+TUWD9gv9/KZlq/zHLccE/AUtmBZ1LWDya2zR8tFv0Q+yzu+YuUTU3sSiuLXUXXM69ZGvSdLa+ZC75QZbn7OzZhtzFQYV4/77vhH+S2POw+94nL4T3XmqFt3HtZ3d+yz+7+/RdRwwb1L9Pr569+vQfNGzErqff85n3fMYg8zdgjObxCw4tbt57IzWJlZ7k6OL0ePZ3H0rG4OgfMkczLdn38bO17bscZRymRPkFWmRfrD2Vpl7Tm+9Hul9/7QfyET2Si9vDpOH0xu4GsuwU+mkNI+o5Uc1xj/dejy3hS4/Y3x1r8cPr07rlbHrBLCYhipnnbkK7FtVj/V9HudfFvF+tW3tXn1a185UvMCl+L1xRc8vClhd+p+lm/StGCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGETtc+A1dbe5tvnnTh+LsfmzTlgzmLlM8+mDLpiXuu+/XJB395xGqD+ub8IQ0iNwZsdfDpVz1scX+SGY/98YxvbR34dE0h2vrtd+V/31nIYqnD4ncn/umgZr69jbDTddju4z9t7D4kbnNv3Hd1+fXY4jY9+R+1j/OJxdIHTw98arNoet3G/ivKs7Ea8PGjh+f48YcifkMOforvfeJeHjeMQUVT6zr2H3zLU/Pvcc1xX2sRZvj/S/h3X5hZ40dQgmg2I2/lm5yRCTsU6jFjIoqd7uG7m6mH96Yc0QxG3LGIb2zmlj7YTI97aGUDftLAu+hJWPHrwZQmCmvHZ/hu5srr+1OeKKJOZ87jG5k7Cy+T90wLarW/8T3MqUda4YEZTWfjJJ5cFLPXdqRYURCj3uBbl3Mzd6dgUQCjXuTbVgDv7UnRIudGPM23rCDeavZHuDWFlep/+FVmXlqT4kVO9TjX/hlhefCXlTkAkUdfMTwIOr/mHMYhiNzp/S++SYX08uochsiXw3P75no0K37GgYgcGfp3vj0FNqnFn3GaQ1+J8tDV3Fv+PQ5H5MPFfGMK7+6eHJHI3rCJfFeawIxNOSiRtR2m8j1pCvPGcVgiWwfl7HzQhv1MrsPPgZP5bjSRq3pzbCIzl/G9aCoT5CdWxq6N704wefKUXCGdpc5XNeeyam9/Qj6Izk6XC/kuNKH7+nOQIm1dTuV70JSu78thipQdxnegSV3SneMUqdqpmKfwRXciByrSNMLipsXFtHw3DlWkp+e/mf0mNk1umpW2Tucx903tPnmJlbJd5zD1TW3paRyuSMfgB5j5JjdlKw5YpKHT0cx70/ujfKCToqEt8VuwbMUYDlmk4FJmvQU8Ij+wUrNWU1wkEdG+HLRI3JXNehZDkGc4aJG0NaYw5S1hobzrnpIfJvIMrty6jcMWyep5PxPeIt5dlwMXidrhXSa8RSw8ngMXiWrCC3D05BdhGnqMZ7pbxnNf4NBFgtZJ7ZmUeTFzLw5dJGj7T5nu1iEnjqZgfya7hfyGQxcJOonJbiG39OPYRWK6X8Rkt5B/y61HE9f3T0x2C3l2PQ5eJGbljJ/SnIXX5L6jiRtwF5PdQt7anIMXiRnYBPc9tjVlJAcvEiPrSiRB1pVIgqwrkQRZVyIJsq5EEmRdiSTIuhJJkHUlkiDrSiRB1pVIgqwrkQRZVyIJsq5EEmRdiSTIuhJJkHUlkiDrSiRB1pVIgqwrkQRZVyIJsq5EFM8wdz5PkFBD1pXL0yT4yM1uQ9fV4yTUkHXlErquniahdcm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPVlXZrKu7Mm6MpN1ZU/WlZmsK3uyrsxkXdmTdWUm68qerCszWVf2ZF2ZybqyJ+vKTNaVPet1dTf7W8jkLTj4WrKuQtmuqwF3sL+FvL4pB19L1lUo23XV7zr2t5DnN+Tga8m6CmW7rnpeyv4W8thaHHwtWVehbNdV24/Y30LuGMCx15J1Fcp6XX2L/S3kik4cey1ZV6Gs19UuC0hoHWdw6D6yrkJZr6uNJpHQMuYexKH7yLoKZb2u+v+NhJbxStjbDLKuwlmvq7azSWgZ93bmyH1kXYX6DzPhM5EEnz0/IqNFLDubA/d7kRSfJ0loXY8wEz4vkOAz6DEyWsSM0Ry43/9I8XmYhNZ1FzPh80pXMnx+uYKU1hD6gqCt7+uk+NxBRuv6MzPh89YQMnxGvUdKS1hyBIftt+YUcnz+REbrOm85U1HrvbCTQ9raWuqUhmldOGq/bWeQU2vZuWS0rh8sZi5qfbofGX47LCSnFZzIQQcYO4ecWou/T0br2mk+c1FryelkBLiHnBbwTn+OOcBZy0iqNX9HMlrX0NnMhc+1ZATYiJQWcBiHHORmcnw+HUxGC3uZufB5eCUy/LpeRE7Te0CzQgY9TpLPJDJa2Z+YC58p25IRYMRbJDW5z77GAQf5yjSyfK4ho5V9h7nwWXE0GUEOI6nJ/S70I5yS40PfxzuEjFa2etgbDe1/7kZKgJ7Xk9TUnludww3S+wayfJYPJ6WV9Qr9JOfVL5ESZEToZxjNY/YeHGygLd4mzeehXqS0sk6nMRt+ur+F2vYJe/Omaaz4MYca7Puk+Z0adn5pS/lS2Duj7XeFndetnEpW07qBAw026J+k+SzW/ZxvHeFXmi4P/yS/7ArSmtTDfTjOYNuR5nen9n/H1nE48+F3beg5DWW9/kpaU3pubQ4zWLfwg/8OKa1uo5eYEL/1SQm2auivguJ7a3MOMsTm5Pm98EVSWt6FzIjfnWSEGBL6t2TRTR3FIYaZQKLfBWSIbULPI/r8q6SEGDiRxCbz0UYcYJh9Q692e3trUkTbeObE7789SAnR5ykSm8oHa3J4YXo/R6afnNLXYdP3mRQf3dkyStcmPGfmhdU4uFBnLyXV572NSREl4TfzeDP8tFFHt1+T2TTuXoVDC7XNO6T6XUKKKOv7IdPid11vckId11Snjy7/veZjUUe/W8j1m6F/06vlhJ7V0N7+XVLCjQlflYWz+CgOSuMYcgN8ixTh6HUnE+O3NPQy8qoBoRdoFs2ssFuouYwiN8BtPckR2OQTpsZvcviJo1UXkFtwN0f4NTZwOsl+n8iLdp+jmJsAj5Kis8NUkgts7uEcjFbonQfa248kRXToHf5iNNJ5tT3+QnJhPWn8O7Ckk+Ywb5LzrgKs+yrT47f4LHK0dp5FeiEt1p5tVnVO6DtX7a98gRzhse9cJshvSaQLLbuEf9CYezdprhN0OWYJ+X5z9yJH1Ag/c7S9PfzqZ7cNC/qxzpSdOQCDA8kPcgo5wudqpijIPuQY7FPAX4aLdVceuX2DBkGuIEf49fkHkxRgedTT1Y4u2m2MLtCevejyPRoEucv4sUQrGx56CW97+8KoP+i7nbWIJgWw4upoL6xKzgi9EKC9/ZFhJIlAG4ZeVt/evvT8qNeZ9Dq/KD+z/hj5XPQuvwy7y0fJS+uRJUJsGnqOX8kfIn+q2u2HNMm1S/pSrlm/8HPU2tvflvfZjTbW3Zb2kci/NNraDsv5K/j5Z+gukq8x8ElaBfkgwoeKYkT4J4Xt7R/ZXBu3/Qu0yqHJ+1JkJCN1czIz7LFMwmPt8LPWSn+TH09WJIMuzecLrVvWoMBoTgl/N7T0S9B00rLAF7Q/Z663+4v6G7m7kcPkH1BaRH1uomGgZ0eQJoxWfZhJC/TuJqRFtNKPP6ZlDsz/Zeh9nkNsEX5eTMmDUT6sFugWelt35RzSIht2ZS7e0lr2V/uPhvWn798W9T1V4dB/hPzKOqRFN/TKjH9qfXZ9HW8ybRh663/lPNJEZF/XvVZtX/xT49UFfkNPyOzkv5ln6G+3EKzH+dpJWLQ3ecLCFm8yfcHeC3v4v1a3bf6p+TQkGUue/Krh6toQW+mvB3ltM/KElZUNz5W4fWUSLfU7+knNRyIxe/qUeu8aNNBwxe0dFu8RC48T9O8+LTqt7hvTDfnW7aH3NY3NfYfr7g+q1/kn+p+ry48hUdRhvbAnwODj3UmsR/f1L5qW0J+Jiz+8fNOGLrna61N6CjFdzjluSB/TTfkm6+/pZ7b1zx/6gL5i8dFjF2xP13X7sunvi8vkbKtGjTE9Fe6lGG7MM2Tr79/U8Or6+NajtzPesSOC7TTnCinTdyJTNKCL9mOMsmdiulFr5x7DD7zoiY/naf+4r7Vk3qyJlx6yVs/wx7nZ2ex5Og51vcV5EEJjK+0nGWWT4r6Z2KqjDzr10hvueeS5N96fXXNbkUWzZ7z1/KN/v+Gy08ZuY/uZjNG24Re7YWpd76+IIF1/bjwtYdrXyS2yA4z/Ay07Uz64idPwR5nYcItOqOMt+Bzpfor5HdtHhpIs4jJGdx4pbizuOSPrRLhp+Afyej0Jx0b4BOadA0gulrGhj3zrsMDytC0RVbfLopz9eZvduZjZWzv83l8dll8S1x+cwm/lG5lmrZmnR7hjVk4MODPS5R3XFeeIimnobcy03tvFeCL7UbpT+TvcIs9rTt4ww2kOFW8flO/bQfUeG/oAQa875I/AdAwJfVJojVdPqPNUmsQNOOU1ajS5dlWaiOT1vYRZN/r4qvxdBrXWNVFPiV7xK/mAOWU/Cr8LW61JBwyiUfYGHRT9urPZJ9NIpGnX8OfL+Sz829ez/wCk2/63W5zv9fwuNBNpW83qqfRzHj0ku096uh3yqNUDqf8sL6sydUToI5uCTT1j47T/TOy98Y8jvJ3uNu0QmorsrDPe9iqIhRN+tgONk7bjz/9l+/SepVfXc0mYiF/nHeu4B8OC6Tfv06/u6y6MOvXb92/T63gg1AvbJVeTsNb5SN392DQ+uvnwkcPjfEnfdfjIcTfXeeutyYfLosqdASeZzoTXmPzglceMqf+qrLLVxxz3+wcn018d3v0/uSAwp/odG+EkLZ0Vn8+a/sz1P9x93V7dupjPIu/cpVuvdfc47YZn3pv1eYP32frgB9FvCSky0PnIWG/St+KTdyY9PuGum8df84fLL7/8D9eMv/muCY9PeueTWG/X9uw4ihe5tv1fkr+WOS5Lx29L1aIA+p/0bqw/UhKxYvJx8pKqeDa7us6/EVMx+YqYrnoUGRh9nuX78amY9vOtqE8UVpfBp7xhdQVzoha/csIqcr1y0+i83436Oyqm4bW/7C3vezah4Xv9Ntr55PGbcvHX5GziptZjrVOemxP+MNu4LZ098cS16rsNpCiezqN/ePvLC/jeJ+Pz/9168pYMJ1rLalt/8zfPxn2z0SUTf3XgVvJbT7R17t5/65Nvee3DOfXfFnLRnA9fufmkLft2lz/2RJBh9vfhnic/nIQQQgghhBBCCCGEEEIIIYQQQgghhGg9bW3/H5J/AZEh38+BAAAAAElFTkSuQmCC';

/**
 * The url of the synthesis server.
 * @type {string}
 */
const SERVER_HOST = 'https://synthesis-service.scratch.mit.edu';

/**
 * The url of the translate server.
 * @type {string}
 */
const serverURL = 'https://translate-service.scratch.mit.edu/';

/**
 * How long to wait in ms before timing out requests to translate server.
 * @type {int}
 */
const serverTimeoutMs = 10000; // 10 seconds (chosen arbitrarily).


/**
 * How long to wait in ms before timing out requests to synthesis server.
 * @type {int}
 */
const SERVER_TIMEOUT = 10000; // 10 seconds

/**
 * Volume for playback of speech sounds, as a percentage.
 * @type {number}
 */
const SPEECH_VOLUME = 250;

/**
 * An id for one of the voices.
 */
const ALTO_ID = 'ALTO';

/**
 * An id for one of the voices.
 */
const TENOR_ID = 'TENOR';

/**
 * An id for one of the voices.
 */
const SQUEAK_ID = 'SQUEAK';

/**
 * An id for one of the voices.
 */
const GIANT_ID = 'GIANT';

/**
 * Playback rate for the tenor voice, for cases where we have only a female gender voice.
 */
const FEMALE_TENOR_RATE = 0.89; // -2 semitones

/**
 * Playback rate for the giant voice, for cases where we have only a female gender voice.
 */
const FEMALE_GIANT_RATE = 0.79; // -4 semitones



/**
 * Class for the motion-related blocks in Scratch 3.0
 * @param {Runtime} runtime - the runtime instantiating this block package.
 * @constructor
 */
class Scratch3TextClassificationBlocks {
    constructor (runtime) {

         /**
         * The result from the most recent translation.
         * @type {string}
         * @private
         */
        this._translateResult = '';

        /**
         * The language of the text most recently translated.
         * @type {string}
         * @private
         */
        this._lastLangTranslated = '';

        /**
         * The text most recently translated.
         * @type {string}
         * @private
         */
        this._lastTextTranslated = '';

        /**
         * The runtime instantiating this block package.
         * @type {Runtime}
         */
         this.scratch_vm = runtime;
         this.sentencesample = [];
         this.labledsample = [];
         this.lastSentenceClassified = null;
         
         this.custom_NLP_model = tf.sequential();


         
        /**
         * The timer utility.
         * @type {Timer}
         */
        this._timer = new Timer();

        /**
         * The stored microphone loudness measurement.
         * @type {number}
         */
        this._cachedLoudness = -1;

        /**
         * The time of the most recent microphone loudness measurement.
         * @type {number}
         */
        this._cachedLoudnessTimestamp = 0;
         
         /**
         * Map of soundPlayers by sound id.
         * @type {Map<string, SoundPlayer>}
         */
        this._soundPlayers = new Map();

        this._stopAllSpeech = this._stopAllSpeech.bind(this);
        if (this.scratch_vm) {
            this.scratch_vm.on('PROJECT_STOP_ALL', this._stopAllSpeech);
        }

        this._onTargetCreated = this._onTargetCreated.bind(this);
        if (this.scratch_vm) {
            this.scratch_vm.on('targetWasCreated', this._onTargetCreated);
        }
        
        this.scratch_vm.on('EDIT_TEXT_MODEL', modelInfo => {
            console.log(modelInfo);
            console.log("Calling bound function");
            this.editModel.bind(this, modelInfo);
        });
        this.scratch_vm.on('EDIT_TEXT_CLASSIFIER', modelInfo => {
            console.log(modelInfo);
            console.log("Calling bound function");
            this.editModel.bind(this, modelInfo);
        });
        
        this.labelList = [''];
        this.labelListEmpty = true;
        
        // When a project is loaded, reset all the model data
        this.scratch_vm.on('PROJECT_LOADED', () => {
            this.clearLocal();
            this.loadModelFromRuntime();
        });
        // Listen for model editing events emitted by the text modal
        this.scratch_vm.on('NEW_EXAMPLES', (examples, label) => {
            this.newExamples(examples, label);
        });
        this.scratch_vm.on('NEW_LABEL', (label) => {
            this.newLabel(label);
        });
        this.scratch_vm.on('DELETE_EXAMPLE', (label, exampleNum) => {
            this.deleteExample(label, exampleNum);
        });
        this.scratch_vm.on('RENAME_LABEL', (oldName, newName) => {
            this.renameLabel(oldName, newName);
        });
        this.scratch_vm.on('DELETE_LABEL', (label) => {
            this.clearAllWithLabel({LABEL: label});
        });
        this.scratch_vm.on('CLEAR_ALL_LABELS', () => {
            if (!this.labelListEmpty && confirm('Are you sure you want to clear all labels?')) {    //confirm with alert dialogue before clearing the model
                let labels = [...this.labelList];
                for (var i = 0; i < labels.length; i++) {
                    this.clearAllWithLabel({LABEL: labels[i]});
                }
                //this.clearAll(); this crashed Scratch for some reason
            }
        });

        //Listen for model editing events emitted by the classifier modal
        this.scratch_vm.on('EXPORT_CLASSIFIER', () => {
            this.exportClassifier();
        });
        this.scratch_vm.on('LOAD_CLASSIFIER', () => {
            console.log("load");
            this.loadClassifier();
            
        });

        this.scratch_vm.on('DONE', () => {
            console.log("DONE");
            this.buildCustomDeepModel();
        });

        
        this._recognizedSpeech = "";

        this._toxicity_labels = {
            items: [
                {
                    value : 'toxicity',
                    text : 'toxic'
                }, {
                    value : 'severe_toxicity',
                    text : 'severely toxic'
                }, {
                    value : 'identity_attack',
                    text : 'an identity-based attack'
                }, {
                    value : 'insult',
                    text : 'insulting'
                }, {
                    value : 'threat',
                    text : 'threatening'
                }, {
                //     value : 'sexual_explicit',
                //     text : 'sexually explicit'
                // }, {
                    value : 'obscene',
                    text : 'profanity'
                }
            ],
            acceptReporters: true
        };
        
        this.resetModelData();
        // load the toxicity model
        this._toxicitymodel = null;
        this._loadToxicity();

        this.sentiment = new Sentiment();

    }

    /**
     * An object with info for each voice.
     */
    get VOICE_INFO () {
        return {
            [SQUEAK_ID]: {
                name: formatMessage({
                    id: 'text2speech.squeak',
                    default: 'squeak',
                    description: 'Name for a funny voice with a high pitch.'
                }),
                gender: 'female',
                playbackRate: 1.19 // +3 semitones
            },
            [TENOR_ID]: {
                name: formatMessage({
                    id: 'text2speech.tenor',
                    default: 'tenor',
                    description: 'Name for a voice with ambiguous gender.'
                }),
                gender: 'male',
                playbackRate: 1
            },
            [ALTO_ID]: {
                name: formatMessage({
                    id: 'text2speech.alto',
                    default: 'alto',
                    description: 'Name for a voice with ambiguous gender.'
                }),
                gender: 'female',
                playbackRate: 1
            },
            [GIANT_ID]: {
                name: formatMessage({
                    id: 'text2speech.giant',
                    default: 'giant',
                    description: 'Name for a funny voice with a low pitch.'
                }),
                gender: 'male',
                playbackRate: 0.84 // -3 semitones
            }
        };
    }
    
     /**
     * The key to load & store a target's text2speech state.
     * @return {string} The key.
     */
    static get STATE_KEY () {
        return 'Scratch.text2speech';
    }

    /**
     * The default state, to be used when a target has no existing state.
     * @type {Text2SpeechState}
     */
    static get DEFAULT_TEXT2SPEECH_STATE () {
        return {
            voiceId: SQUEAK_ID
        };
    }
    
    /**
     * @param {Target} target - collect  state for this target.
     * @returns {Text2SpeechState} the mutable state associated with that target. This will be created if necessary.
     * @private
     */
    _getState (target) {
        let state = target.getCustomState(Scratch3TextClassificationBlocks.STATE_KEY);
        if (!state) {
            state = Clone.simple(Scratch3TextClassificationBlocks.DEFAULT_TEXT2SPEECH_STATE);
            target.setCustomState(Scratch3TextClassificationBlocks.STATE_KEY, state);
        }
        return state;
    }

    /**
     * When a Target is cloned, clone the state.
     * @param {Target} newTarget - the newly created target.
     * @param {Target} [sourceTarget] - the target used as a source for the new clone, if any.
     * @listens Runtime#event:targetWasCreated
     * @private
     */
    _onTargetCreated (newTarget, sourceTarget) {
        if (sourceTarget) {
            const state = sourceTarget.getCustomState(Scratch3TextClassificationBlocks.STATE_KEY);
            if (state) {
                newTarget.setCustomState(Scratch3TextClassificationBlocks.STATE_KEY, Clone.simple(state));
            }
        }
    }
    
    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        // Set the video display properties to defaults the first time
        // getInfo is run. This turns on the video device when it is
        // first added to a project, and is overwritten by a PROJECT_LOADED
        // event listener that later calls updateVideoDisplay
        if (this.firstInstall) {
            this.globalVideoState = VideoState.OFF;
            this.globalVideoTransparency = 50;
            this.updateVideoDisplay();
            this.firstInstall = false;
            this.predictionState = {};
        }

        // Return extension definition
        return {
            id: 'speechRecognition',
            name: formatMessage({
                id: 'speechRecognition',
                default: '음성 인식',
                description: ''
            }),
            blockIconURI: blockIconURI,
            menuIconURI: menuIconURI,
            //color1, color2, color3
            blocks: [
                {
                    opcode: 'speakText',
                    text: formatMessage({
                        id: 'textClassification.speakText',
                        default: '[TEXT] 말하기',
                        description: 'Send text to the speech to text engine'
                    }),
                    blockType: BlockType.COMMAND,
                    arguments: {
                        TEXT: {
                            type: ArgumentType.STRING,
                            defaultValue: 'Hello'
                        }
                    },
                },
                {
                    opcode: 'askSpeechRecognition',
                    text: formatMessage({
                        id: 'textClassification.askSpeechRecognition',
                        default: '[PROMPT]를 말하고 응답 기다리기',
                        description: 'Get the class name that the input text matches'
                    }),
                    blockType: BlockType.COMMAND,
                    arguments: {
                        PROMPT: {
                            type: ArgumentType.STRING,
                            defaultValue: 'How are you?'
                        }
                    },
                },
                {
                    opcode: 'getRecognizedSpeech',
                    text: formatMessage({
                        id: 'textClassification.getRecognizedSpeech',
                        default: '응답',
                        description: 'Return the results of the speech recognition'
                    }),
                    blockType: BlockType.REPORTER,
                },
                {
                    opcode: 'setVoice',
                    text: formatMessage({
                        id: 'text2speech.setVoiceBlock',
                        default: '음성 [VOICE]로 설정하기',
                        description: 'Set the voice for speech synthesis.'
                    }),
                    blockType: BlockType.COMMAND,
                    arguments: {
                        VOICE: {
                            type: ArgumentType.STRING,
                            menu: 'voices',
                            defaultValue: SQUEAK_ID
                        }
                    }
                },
                {
                    opcode: 'onHeardSound',
                    text: formatMessage({
                        id: 'textClassification.onHeardSound',
                        default: '소리가 [THRESHOLD]보다 클 때',
                        description: 'Event that triggers when a sound is heard above a threshold'
                    }),
                    blockType: BlockType.HAT,
                    arguments: {
                        THRESHOLD: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 10
                        }
                    },
                }
            ],
            menus: {
                voices: {
                    acceptReporters: true,
                    items: this.getVoiceMenu()
                },
                model_classes: {
                    acceptReporters: false,
                    items: 'getLabels'
                },
                toxicitylabels : this._toxicity_labels
            }
        };
    }
    
    /**
     * TODO Moves info from the runtime into the classifier, called when a project is loaded
     */    
    async loadModelFromRuntime () {
        //console.log("Load model from runtime");
        this.labelList = [];
        this.labelListEmpty = false;
        let textData = this.scratch_vm.modelData.textData;

        for (let label in this.scratch_vm.modelData.textData) {
            if (this.scratch_vm.modelData.textData.hasOwnProperty(label)) {
                let textExamples = textData[label];
                this.newLabel(label);
                this.newExamples(textExamples, label);
            }
        }

        if (this.labelList.length == 0) {
            this.labelList.push('');    //if the label list is empty, fill it with an empty string
            this.labelListEmpty = true;
        }
        /*console.log("RANDI try a practice class");
        this.scratch_vm.modelData = {textData: {'Class 1':['Example 1','Example 2','Here\'s a really long example to make sure things are working','Example 3','Example 4','Example 5','Example 6']}, classifierData: {'Class 1':['Example 1','Example 2','Here\'s a really long example to make sure things are working','Example 3','Example 4','Example 5','Example 6']}, nextLabelNumber: 2};
        this.labelList = ['Class 1'];
        this.labelListEmpty = false;*/

        await this.buildCustomDeepModel()

    }

    /**
     * Return label list for block menus
     * @return {array of strings} an array of the labels for the text model classifier
     */
    getLabels () {
        return this.labelList;
    }

    /**
     * TODO grab text and add it as an example
     * @param {string} args.LABEL the name of the label to add an example to
     */
    textExample (args) {
        console.log("Get text example");
        // TODO grab text
        let text = '';
         if (frame) {
             this.newExamples([text], args.LABEL);
         }
    }

    /**
     * TODO Add new examples to a label
     * @param {array of strings} examples a list of text examples to add to a label
     * @param {string} label the name of the label
     */
    newExamples (text_examples, label) {   //add examples for a label
        console.log("Add examples to label " + label);
        console.log(text_examples);
        if (this.labelListEmpty) {
            // Edit label list accordingly
            this.labelList.splice(this.labelList.indexOf(''), 1);
            this.labelListEmpty = false;
        }
        if (!this.labelList.includes(label)) {
            this.labelList.push(label);
        }
        for (let text_example of text_examples) {
            if (!this.scratch_vm.modelData.textData[label].includes(text_example)) {
                this.scratch_vm.modelData.textData[label].push(text_example);
                this.scratch_vm.modelData.classifierData[label].push(text_example);
            }
        }

    }
    
    /**
     * TODO Add a new label to labelList
     * @param {string} label the name of the label
     */
    newLabel (newLabelName) {   //add the name of a new label
        if (this.labelListEmpty) {
            // Edit label list accordingly
            this.labelList.splice(this.labelList.indexOf(''), 1);
            this.labelListEmpty = false;
        }
        if (!this.labelList.includes(newLabelName)) {
            this.labelList.push(newLabelName);
        }
        
        this.scratch_vm.modelData.textData[newLabelName] = [];
        this.scratch_vm.modelData.classifierData[newLabelName] = [];
        // update drowndown of class names
        //this.scratch_vm.emit("TOOLBOX_EXTENSIONS_NEED_UPDATE");
        this.scratch_vm.requestToolboxExtensionsUpdate();
    }
    



    /**
     * TODO Rename a label
     * @param {string} oldName the name of the label to change
     * @param {string} newname the new name for the label
     */
    renameLabel (oldName, newName) {
        console.log("Rename a label");

        this.scratch_vm.modelData.classifierData[newName] = this.scratch_vm.modelData.classifierData[oldName];  //reset the runtime's model data with the new renamed label (to share with GUI)
        delete this.scratch_vm.modelData.classifierData[oldName];

        this.scratch_vm.modelData.textData[newName] = this.scratch_vm.modelData.textData[oldName];  //reset the runtime's model data with the new renamed label (to share with GUI)
        delete this.scratch_vm.modelData.textData[oldName];

        this.labelList.splice(this.labelList.indexOf(oldName), 1);  //reset label list with the new renamed label
        this.labelList.push(newName);
    }

    /**
     * TODO Delete an example (or all loaded examples, if exampleNum === -1)
     * @param {string} label the name of the label with the example to be removed
     * @param {integer} exampleNum which example, in the array of a label's examples, to remove
     */
    deleteExample (label, exampleNum) {
        console.log("Delete example " + exampleNum + " with label " + label);
         // Remove label from the runtime's model data (to share with the GUI)
         if (exampleNum === -1) {    //if this is true, delete all the loaded examples
            let numLoadedExamples = this.scratch_vm.modelData.classifierData[label].length - this.scratch_vm.modelData.textData[label].length;   //imageData[label].length is ONLY the length of the NEW examples (not the saved and then loaded ones!)
            this.scratch_vm.modelData.classifierData[label].splice(0, numLoadedExamples);
         } else {
         this.scratch_vm.modelData.textData[label].splice(exampleNum, 1);
         this.scratch_vm.modelData.classifierData[label].splice(exampleNum - this.scratch_vm.modelData.textData[label].length - 1, 1);
         }
    }

    /**
     * TODO Clear all data stored in the classifier and label list
     */
    clearLocal () {
        console.log("Clear local data");
        this.scratch_vm.emit("TOOLBOX_EXTENSIONS_NEED_UPDATE");
        this.labelList = [''];
        this.labelListEmpty = true;
    }

    resetModelData() {
        this.scratch_vm.modelData = {textData: {}, classifierData: {}, nextLabelNumber: 1};
    }

    /**
     * TODO Clear local label list, but also clear all data stored in the runtime
     */
    clearAll () {
        console.log("Clear all data");
        this.clearLocal();
        // Clear runtime's model data
        
        this.resetModelData();
    }

    /**
     * TODO Clear all examples with a given label
     * @param {string} args.LABEL the name of the label to remove from the model
     */
    clearAllWithLabel (args) {
        console.log("Get rid of all examples with label " + args.LABEL);
        if (this.labelList.includes(args.LABEL)) {
            // Remove label from labelList
            this.labelList.splice(this.labelList.indexOf(args.LABEL), 1);
            // Remove label from the runtime's model data (to share with the GUI)
            delete this.scratch_vm.modelData.classifierData[args.LABEL];  
            delete this.scratch_vm.modelData.textData[args.LABEL];
            // If the label list is now empty, fill it with an empty string
            if (this.labelList.length === 0) {  
                this.labelListEmpty = true;
                this.labelList.push('');
            }
        }
    }
    
    
    /**
     * Detects if the sound from the input mic is louder than a particular threshold
     * @param args.THRESHOLD {integer} the threshold of loudness to trigger on
     * @return {integer} true if the loudness is above the threshold and false if it is not
     */    
    onHeardSound(args) {
        let threshold = args.THRESHOLD;
        
        return this.getLoudness() > threshold;
    }
    
    /**
     * Get the input volume from the mic
     * @return {integer} mic volume at current time
     */
    getLoudness () {
        if (typeof this.scratch_vm.audioEngine === 'undefined') return -1;
        if (this.scratch_vm.currentStepTime === null) return -1;

        // Only measure loudness once per step
        const timeSinceLoudness = this._timer.time() - this._cachedLoudnessTimestamp;
        if (timeSinceLoudness < this.scratch_vm.currentStepTime) {
            return this._cachedLoudness;
        }

        this._cachedLoudnessTimestamp = this._timer.time();
        this._cachedLoudness = this.scratch_vm.audioEngine.getLoudness();
        return this._cachedLoudness;
    }
    
    /**
     * Get the menu of voices for the "set voice" block.
     * @return {array} the text and value for each menu item.
     */
    getVoiceMenu () {
        return Object.keys(this.VOICE_INFO).map(voiceId => ({
            text: this.VOICE_INFO[voiceId].name,
            value: voiceId
        }));
    }
    
    /**
     * Set the voice for speech synthesis for this sprite.
     * @param  {object} args Block arguments
     * @param {object} util Utility object provided by the runtime.
     */
    setVoice (args, util) {
        const state = this._getState(util.target);

        let voice = args.VOICE;

        // If the arg is a dropped number, treat it as a voice index
        let voiceNum = parseInt(voice, 10);
        if (!isNaN(voiceNum)) {
            voiceNum -= 1; // Treat dropped args as one-indexed
            voiceNum = MathUtil.wrapClamp(voiceNum, 0, Object.keys(this.VOICE_INFO).length - 1);
            voice = Object.keys(this.VOICE_INFO)[voiceNum];
        }

        // Only set the voice if the arg is a valid voice id.
        if (Object.keys(this.VOICE_INFO).includes(voice)) {
            state.voiceId = voice;
        }
    }
    
    /**
     * Stop all currently playing speech sounds.
     */
    _stopAllSpeech () {
        this._soundPlayers.forEach(player => {
            player.stop();
        });
    }

    /**
     * Convert the provided text into a sound file and then play the file.
     * @param  {object} args Block arguments
     * @param {object} util Utility object provided by the runtime.
     * @return {Promise} A promise that resolves after playing the sound
     */
    async speakText(args, util) {
        // Cast input to string
        let words = Cast.toString(args.TEXT);
        let locale = 'en-US';

        const state = this._getState(util.target);

        let gender = this.VOICE_INFO[state.voiceId].gender;
        let playbackRate = this.VOICE_INFO[state.voiceId].playbackRate;
        
        // Build up URL
        let path = `${SERVER_HOST}/synth`;
        path += `?locale=${locale}`;
        path += `&gender=${gender}`;
        path += `&text=${encodeURIComponent(words.substring(0, 128))}`;
        // Perform HTTP request to get audio file
        return new Promise(resolve => {
            nets({
                url: path,
                timeout: SERVER_TIMEOUT
            }, (err, res, body) => {
                if (err) {
                    console.warn(err);
                    return resolve();
                }

                if (res.statusCode !== 200) {
                    console.warn(res.statusCode);
                    return resolve();
                }

                // Play the sound
                const sound = {
                    data: {
                        buffer: body.buffer
                    }
                };
                this.scratch_vm.audioEngine.decodeSoundPlayer(sound).then(soundPlayer => {
                    this._soundPlayers.set(soundPlayer.id, soundPlayer);

                    soundPlayer.setPlaybackRate(playbackRate);

                    // Increase the volume
                    const engine = this.scratch_vm.audioEngine;
                    const chain = engine.createEffectChain();
                    chain.set('volume', SPEECH_VOLUME);
                    soundPlayer.connect(chain);

                    soundPlayer.play();
                    soundPlayer.on('stop', () => {
                        this._soundPlayers.delete(soundPlayer.id);
                        resolve();
                    });
                });
            });
        });
    }
    
    recognizeSpeech() {
        let recognition = new webkitSpeechRecognition();
        let self = this;
        
        return new Promise(resolve => {
            recognition.start();
            recognition.onresult = function(event) {
                if (event.results.length > 0) {
                    self._recognizedSpeech = event.results[0][0].transcript;
                }
                resolve();
            };
        });
    }
    
    async askSpeechRecognition(args, util) {
        let prompt = Cast.toString(args.PROMPT);
        args.TEXT = prompt;
        let speakTextResolved = await this.speakText(args, util);
        return this.recognizeSpeech();
     }
    
    getRecognizedSpeech() {
        return this._recognizedSpeech;
    }

    /**
     * A scratch conditional block that checks if a text example is a part of a particular class
     * @param {object} args - the block arguments
     * @param {BlockUtility} util - the block utility
     * @returns {boolean} true if the model matches
     *   reference
     */
    async ifTextMatchesClass(args, util) {
        const text = args.TEXT;
        const className = args.CLASS_NAME;
        const predictionState = await this.get_embeddings(text,"none","predict");
        
        if (!predictionState) {
            return false;
        } else {
            const currentMaxClass = predictionState;
            return (currentMaxClass == String(className));
        }
    }

    /**
     * A scratch hat block reporter that returns whether the input text matches the model class.
     * @param {object} args - the block arguments
     * @param {BlockUtility} util - the block utility
     * @returns {string} class name if input text matched, empty string if there's a problem with the model
     */
    async getModelPrediction(args) {
        const text = args.TEXT;
        const predictionState = await this.get_embeddings(text,"none","predict");
        return predictionState;
    }

    /**
     * A scratch hat block reporter that returns the confidance level of the input text matches the model class.
     * @param {object} args - the block arguments
     * @param {BlockUtility} util - the block utility
     * @returns {float} confidence of class name if input text matched, empty string if there's a problem with the model
     */
    async getModelConfidence(args) {
        const text = args.TEXT;
        const predictionConfidence = await this.get_confidence(text,"none","predict");
        return predictionConfidence;
    }
    /**
     * Returns whether or not the text inputted is one of the examples inputted
     * @param text - the text inputted
     * @param className - the class whose examples are being checked
     * @returns a boolean true if the text is an example or false if the text is not an example
     */
    getPredictedClass(text,className) { 
        if (!this.labelListEmpty) {   //whenever the classifier has some data
            try {
            for (let example of this.scratch_vm.modelData.textData[className]) {
                if (text.toLowerCase() === example.toLowerCase()) {
                    return true;
                }
            }
        } catch(err) {
            return false;
        }
        } else { //if there is no data in the classifier
            return false;
        }
    

    }

    /**
     * Calls the method which embeds text as a 2d tensor
     * @param text - the text inputted
     * @param label - this is always "none" when embedding examples
     * @param direction - is either "example" when an example is being inputted or "predict" when a word to be classified is inputted
     */
        // async getembeddedwords(text,label,direction) {
        //     if (!this.labelListEmpty) {
        //         const embeddedtext = await this.get_embeddings(text,label,direction);
        //     }
        // }


    /**
     * Embeds text and either adds examples to classifier or returns the predicted label
     * @param text - the text inputted
     * @param label - the label to add the example to
     * @param direction - is either "example" when an example is being inputted or "predict" when a word to be classified is inputted
     * @returns if the direction is "predict" returns the predicted label for the text inputted
     */
    async get_embeddings(text,label,direction) { //changes text into a 2d tensor
        const newText = await this.getTranslate(text,"en"); //translates text from any language to english
        console.log(newText);

        if (!this.labelListEmpty) { 
            if (direction === "predict") {
                await this.predictScore(newText);
                return this.predictionClass;
            }
            } else {
                return "No classes inputted";
                
        }
    }

    /**
     * Embeds text and either adds examples to classifier or returns the predicted label
     * @param text - the text inputted
     * @param label - the label to add the example to
     * @param direction - is eitfher "example" when an example is being inputted or "predict" when a word to be classified is inputted
     * @returns if the direction is "predict" returns the predicted label for the text inputted
     */
    async get_confidence(text,label,direction) { //changes text into a 2d tensor
        const newText = await this.getTranslate(text,"en"); //translates text from any language to english
        console.log(newText);

        if (direction === "predict") {
            await this.predictScore(newText);
            return this.predictionScore;
        } else {
            return "No classes inputted";
            
        }
    }
    

    /**
     * Embeds text and either adds examples to classifier or returns the predicted label
     * @param text - the text inputted
     * @param label - the label to add the example to
     * @param direction - is either "example" when an example is being inputted or "predict" when a word to be classified is inputted
     * @returns if the direction is "predict" returns the predicted label for the text inputted
     */
    async buildCustomDeepModel() { 
        var numClass = this.labelList.length;

        if (numClass < 2){
            return "No classes inputted";
        }

        //console.log(this.scratch_vm)
        this.scratch_vm.emit('SAY', this.scratch_vm.executableTargets[1], 'think', 'wait .. loading model');

        // console.log(this.labelList)
        this.custom_NLP_model = tf.sequential();
        this.sentencesample = [];
        this.labledsample = [];
        for (let label of this.labelList){
            //console.log(this.scratch_vm.modelData.textData[label])
            for (let sentense of this.scratch_vm.modelData.textData[label]){
                //console.log(sentense)
                this.sentencesample.push(sentense);
                this.labledsample.push(label);
            }
        }
        console.log(this.labledsample)
        console.log(this.sentencesample)


        // console.log('model::'+JSON.stringify(this.custom_NLP_model));
        const ys = tf.oneHot(tf.tensor1d(this.labledsample.map((a) => 
            this.labelList.findIndex(e => e === a)), 'int32'), numClass);
        // console.log('ys',ys);

        const trainingData = await use.load()
        .then(model => {
            return model.embed(this.sentencesample)
                .then(embeddings => {
                    return embeddings;
                });
        })
        .catch(err => console.error('Fit Error:', err));
        // console.log(trainingData)


        // Add layers to the model
        // trying: layers: 3-1 , activation: sigmid/softmax, kernelInitializer: ones/not, 
        // optimozation: adam/sgd, loss function: MSE/CC
        // this.custom_NLP_model.add(tf.layers.dense({
        //     inputShape: [512], //Universal Sentence Encoder - 512-dimensional embedding
        //     activation: 'sigmoid',
        //     units: 128,//aribatry 
        // }));

        // this.custom_NLP_model.add(tf.layers.dense({
        //     inputShape: [128],
        //     activation: 'sigmoid',
        //     units: 32,//aribatry 
        // }));

        // this.custom_NLP_model.add(tf.layers.dense({ 
        //     inputShape: [32],
        //     activation: 'sigmoid',
        //     units: numClass,//number of label classes
        // }));

        // Add layers to the model
        // this.custom_NLP_model.add(tf.layers.dense({
        //     inputShape: [512], //Universal Sentence Encoder - 512-dimensional embedding
        //     activation: 'softmax',
        //     kernelInitializer: 'ones',
        //     units: 128,//aribatry 
        // }));

        // this.custom_NLP_model.add(tf.layers.dense({
        //     inputShape: [128],
        //     activation: 'softmax',
        //     kernelInitializer: 'ones',
        //     units: 32,//aribatry 
        // }));

        // this.custom_NLP_model.add(tf.layers.dense({ 
        //     inputShape: [32],
        //     activation: 'softmax',
        //     kernelInitializer: 'ones',
        //     units: numClass,//number of label classes
        // }));
        // Add layers to the model
        // this.custom_NLP_model.add(tf.layers.dense({
        //     inputShape: [512], 
        //     activation: 'softmax',
        //     kernelInitializer: 'ones',
        //     units: numClass,//number of label classes
        // }));

        // Add layers to the model
        this.custom_NLP_model.add(tf.layers.dense({
            inputShape: [512], 
            activation: 'sigmoid',
            kernelInitializer: 'ones',
            units: numClass,//number of label classes
        }));

        // Compile the model
        this.custom_NLP_model.compile({
            loss: 'meanSquaredError',
            optimizer: tf.train.adam(.06), // This is a standard compile config
        });
        // this.custom_NLP_model.compile({
        //     loss: 'categoricalCrossentropy', 
        //     optimizer: 'sgd', 
        //     metrics: ['acc']
        // });


        await this.custom_NLP_model.fit(trainingData, ys, {
            epochs: 100,
            batchSize: 4,
            shuffle: true,
            validationSplit: 0.15,
            doValidation: true,
            callbacks: [
                tf.callbacks.earlyStopping({monitor: 'val_loss', patience:50})
            ]
         }).then(info => {
           console.log('Final accuracy', info);
         });


        // // This is LSTM exprement - wasn't good
        // const window_size = 512; 
        // const n_epochs = 500; 
        // const learning_rate = 0.0001;
        // const  n_layers = 1;

        // const input_layer_shape  = window_size;
        // const input_layer_neurons = 100;

        // const rnn_input_layer_features = 10;
        // const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;

        // const rnn_input_shape  = [rnn_input_layer_features, rnn_input_layer_timesteps];
        // const rnn_output_neurons = 20;

        // const rnn_batch_size = 4;

        // const output_layer_shape = rnn_output_neurons;
        // const output_layer_neurons = numClass;


        // this.custom_NLP_model.add(tf.layers.dense({units: input_layer_neurons, inputShape: [input_layer_shape]}));
        // this.custom_NLP_model.add(tf.layers.reshape({targetShape: rnn_input_shape}));

        // let lstm_cells = [];
        // for (let index = 0; index < n_layers; index++) {
        //    lstm_cells.push(tf.layers.lstmCell({units: rnn_output_neurons}));
        // }

        // this.custom_NLP_model.add(tf.layers.rnn({
        //     cell: lstm_cells,
        //     inputShape: rnn_input_shape,
        //     returnSequences: false
        // }));

        // this.custom_NLP_model.add(tf.layers.dense({units: output_layer_neurons, inputShape: [output_layer_shape]}));

        // this.custom_NLP_model.compile({
        //     optimizer: tf.train.adam(learning_rate),
        //     loss: 'meanSquaredError'
        // });

        // const hist = await this.custom_NLP_model.fit(trainingData, ys,{ 
        //     batchSize: rnn_batch_size, 
        //     epochs: n_epochs,
        //     validationSplit: 0.15, 
        //     doValidation: true,
        //     shuffle: true,
        //     callbacks: [
        //         tf.callbacks.earlyStopping({monitor: 'val_loss', patience:100})
        //     ]
        // });
        // console.log('Final accuracy: ', hist);

        console.log('model is trained')
        this.scratch_vm.emit('SAY', this.scratch_vm.executableTargets[1], 'say', 'The model is ready');
        
    }

    /**
     * Embeds text and either adds examples to classifier or returns the predicted label
     * @param text - the text inputted
     * @param label - the label to add the example to
     * @param direction - is either "example" when an example is being inputted or "predict" when a word to be classified is inputted
     * @returns if the direction is "predict" returns the predicted label for the text inputted
     */
    async predictScore(text) { 
        // TODO: check if already got the predection
        if (this.lastSentenceClassified === text){
            console.log('Classification done')
            console.log('Predicted Label', this.predictionLabel);
            console.log('Predicted Class', this.predictionClass );
            console.log('Predicted Score', this.predictionScore );
            return "elready done."
        }
        else{
            this.lastSentenceClassified = text;
        }

        // TODO: early stopping  
        // var test_ex = ['great','nope','yes','no']
        //var test_ex = ["sorry , that is not true","yes sir","i am uncertain","please say it again"]
        //var test_ex = ["i really do not have a clue"]
        console.log(text)
        var test_ex = [text]
        const testData = await use.load()
        .then(model => {
            return model.embed(test_ex)
                .then(embeddings => {
                    return embeddings;
                });
        })
        .catch(err => console.error('Fit Error:', err));
        // console.log(testData)
        // this.output = await this.custom_NLP_model.predict(testData);
        await this.custom_NLP_model.predict(testData).print();
        // console.log(this.output);

        const prediction = await this.custom_NLP_model.predict(testData);
        const predict = await prediction.data()
        this.predictionLabel = await prediction.as1D().argMax().dataSync()[0];
        this.predictionClass = this.labelList[this.predictionLabel];
        this.predictionScore = predict[this.predictionLabel];

        console.log('Predicted Label', this.predictionLabel);
        console.log('Predicted Class', this.predictionClass );
        console.log('Predicted Score', this.predictionScore );


    }
   /**
     * Exports the labels and examples in the form of a json document with the default name of "classifier-info.json"
     */
      exportClassifier() { //exports classifier as JSON file
        let dataset = this.scratch_vm.modelData.textData;
        let jsonStr = JSON.stringify(dataset);
        //exports json file
        var data = "text/json;charset=utf-8," + encodeURIComponent(jsonStr);
        var a = document.createElement('a');
        a.setAttribute("href", "data:" + data);
        a.setAttribute("download", "classifier-info.json");
        a.click();
      }
   /**
     * Loads the json document which contains labels and examples. Inputs the labels and examples into the classifier
     */
      async loadClassifier() { //loads classifier to project
        var self = this
        var dataset = document.getElementById("imported-classifier").files[0];
        if (dataset !== undefined) {
            fr = new FileReader();
            fr.onload = receivedText;
            await fr.readAsText(dataset);
            await this.buildCustomDeepModel();
        }
  
          function receivedText(e) { //parses through the json document and adds to the model textData and classifier
            let lines = e.target.result;
            try {
                var newArr = JSON.parse(lines);
                self.clearAll();
                for (let label in newArr) {
                    if (newArr.hasOwnProperty(label)) {
                        let textExamples = newArr[label];
                        self.newLabel(label);
                        self.newExamples(textExamples, label);
                    }
                }
            } catch (err) {
                console.log("Incorrect document form");
            }
                
        }
    }

    getTranslate (words,language) {
        // Don't remake the request if we already have the value.
        if (this._lastTextTranslated === words &&
            this._lastLangTranslated === language) {
            return this._translateResult;
        }

        const lang = language;

        let urlBase = `${serverURL}translate?language=`;
        urlBase += lang;
        urlBase += '&text=';
        urlBase += encodeURIComponent(words);

        const tempThis = this;
        const translatePromise = new Promise(resolve => {
            nets({
                url: urlBase,
                timeout: serverTimeoutMs
            }, (err, res, body) => {
                if (err) {
                    log.warn(`error fetching translate result! ${res}`);
                    resolve('');
                    return '';
                }
                const translated = JSON.parse(body).result;
                tempThis._translateResult = translated;
                // Cache what we just translated so we don't keep making the
                // same call over and over.
                tempThis._lastTextTranslated = words;
                tempThis._lastLangTranslated = language;
                resolve(translated);
                return translated;
            });

        });
        translatePromise.then(translatedText => translatedText);
        return translatePromise;
    }
     

    confidenceTrue(args) {
        return this._classifyText(args.TEXT, args.LABEL, true);
    }
    confidenceFalse(args) {
        return this._classifyText(args.TEXT, args.LABEL, false);
    }

    //-----------------------------------------------------------------------

    _loadToxicity() {
        var id = 'script-toxicity';
        if (document.getElementById(id)) {
            console.log('Toxicity script already loaded');
        }
        else {
            console.log('loading Toxicity script');

            var scriptObj = document.createElement('script');
            scriptObj.id = id;
            scriptObj.type = 'text/javascript';
            scriptObj.src = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/toxicity';

            scriptObj.onreadystatechange = this._loadToxicityModel.bind(this);
            scriptObj.onload = this._loadToxicityModel.bind(this);

            document.head.appendChild(scriptObj);
        }
    }

    _loadToxicityModel() {
        if (this._toxicitymodel) {
            console.log('Toxicity model already loaded');
        }
        else {
            console.log('loading Toxicity model');

            const threshold = 0.1;
            toxicity.load(threshold, this._toxicity_labels.items.map(lbl => lbl.value))
                .then((model) => {
                    console.log('loaded Toxicity model');
                    this._toxicitymodel = model;
                })
                .catch((err) => {
                    console.log('Failed to load toxicity model', err);
                });
        }
    }

    //-----------------------------------------------------------------------

    _classifyText(text, label, returnPositive) {
        if (this._toxicitymodel && text && label) {
            return this._toxicitymodel.classify([ text ])
                .then((predictions) => {
                    const filtered = predictions.filter(prediction => prediction.label === label);
                    if (filtered && filtered.length === 1 &&
                        filtered[0].results && filtered[0].results.length > 0)
                    {
                        const idx = returnPositive ? 1 : 0;
                        return Math.round(filtered[0].results[0].probabilities[idx] * 100);
                    }
                    return 0;
                })
                .catch((err) => {
                    console.log('Failed to classify text', err);
                });
        }
    }

    
    /**
     * chech sentiment score of a text 
     * @param text - the text inputted
     * @returns predicted acore for the text inputted -5 (negative) to 5 (positive)
     */
    async sentimentScore(text){
        console.log(text.TEXT);
        console.log(this.sentiment.analyze(text.TEXT))
        this.predictedsentimentScore = this.sentiment.analyze(text.TEXT)
        return this.predictedsentimentScore.comparative;
    }

      
}

module.exports = Scratch3TextClassificationBlocks;
