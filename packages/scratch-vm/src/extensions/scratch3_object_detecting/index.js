require("regenerator-runtime/runtime");
const Runtime = require('../../engine/runtime');

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Cast = require('../../util/cast');
const formatMessage = require('format-message');
const Video = require('../../io/video');

require("@tensorflow/tfjs-backend-cpu");
require("@tensorflow/tfjs-backend-webgl");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const tf = require("@tensorflow/tfjs");
const { isLabeledStatement } = require("typescript");
const { tidy } = require("@tensorflow/tfjs");

function friendlyRound(amount) {
    return Number(amount).toFixed(2);
}

/**
 * Icon svg to be displayed in the blocks category menu, encoded as a data URI.
 * @type {string}
 */
// eslint-disable-next-line max-len
const menuIconURI = 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMjBweCIgaGVpZ2h0PSIyMHB4IiB2aWV3Qm94PSIwIDAgMjAgMjAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDUyLjIgKDY3MTQ1KSAtIGh0dHA6Ly93d3cuYm9oZW1pYW5jb2RpbmcuY29tL3NrZXRjaCAtLT4KICAgIDx0aXRsZT5FeHRlbnNpb25zL1NvZnR3YXJlL1ZpZGVvLVNlbnNpbmctTWVudTwvdGl0bGU+CiAgICA8ZGVzYz5DcmVhdGVkIHdpdGggU2tldGNoLjwvZGVzYz4KICAgIDxnIGlkPSJFeHRlbnNpb25zL1NvZnR3YXJlL1ZpZGVvLVNlbnNpbmctTWVudSIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InZpZGVvLW1vdGlvbiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDUuMDAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPGNpcmNsZSBpZD0iT3ZhbC1Db3B5IiBmaWxsPSIjMEVCRDhDIiBvcGFjaXR5PSIwLjI1IiBjeD0iMTYiIGN5PSI4IiByPSIyIj48L2NpcmNsZT4KICAgICAgICAgICAgPGNpcmNsZSBpZD0iT3ZhbC1Db3B5IiBmaWxsPSIjMEVCRDhDIiBvcGFjaXR5PSIwLjUiIGN4PSIxNiIgY3k9IjYiIHI9IjIiPjwvY2lyY2xlPgogICAgICAgICAgICA8Y2lyY2xlIGlkPSJPdmFsLUNvcHkiIGZpbGw9IiMwRUJEOEMiIG9wYWNpdHk9IjAuNzUiIGN4PSIxNiIgY3k9IjQiIHI9IjIiPjwvY2lyY2xlPgogICAgICAgICAgICA8Y2lyY2xlIGlkPSJPdmFsIiBmaWxsPSIjMEVCRDhDIiBjeD0iMTYiIGN5PSIyIiByPSIyIj48L2NpcmNsZT4KICAgICAgICAgICAgPHBhdGggZD0iTTExLjMzNTk3MzksMi4yMDk3ODgyNSBMOC4yNSw0LjIwOTk1NjQ5IEw4LjI1LDMuMDUgQzguMjUsMi4wNDQ4ODIyNyA3LjQ2ODU5MDMxLDEuMjUgNi41LDEuMjUgTDIuMDUsMS4yNSBDMS4wMzgwNzExOSwxLjI1IDAuMjUsMi4wMzgwNzExOSAwLjI1LDMuMDUgTDAuMjUsNyBDMC4yNSw3Ljk2MzY5OTM3IDEuMDQyMjQ5MTksOC43NTU5NDg1NiAyLjA1LDguOCBMNi41LDguOCBDNy40NTA4MzAwOSw4LjggOC4yNSw3Ljk3MzI3MjUgOC4yNSw3IEw4LjI1LDUuODU4NDUyNDEgTDguNjI4NjIzOTQsNi4wODU2MjY3NyBMMTEuNDI2Nzc2Nyw3Ljc3MzIyMzMgQzExLjQzNjg5NDMsNy43ODMzNDA5MSAxMS40NzU3NjU1LDcuOCAxMS41LDcuOCBDMTEuNjMzNDkzMiw3LjggMTEuNzUsNy42OTEyNjAzNCAxMS43NSw3LjU1IEwxMS43NSwyLjQgQzExLjc1LDIuNDE4MzgyNjkgMTEuNzIxOTAyOSwyLjM1MjgyMjgyIDExLjY4NTYyNjgsMi4yNzg2MjM5NCBDMTEuNjEyOTUyOCwyLjE1NzUwMDY5IDExLjQ3MDc5NjgsMi4xMjkwNjk1IDExLjMzNTk3MzksMi4yMDk3ODgyNSBaIiBpZD0idmlkZW9fMzdfIiBzdHJva2Utb3BhY2l0eT0iMC4xNSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNSIgZmlsbD0iIzRENEQ0RCI+PC9wYXRoPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+';

/**
 * Icon svg to be displayed at the left edge of each extension block, encoded as a data URI.
 * @type {string}
 */
// eslint-disable-next-line max-len
const blockIconURI = 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iNDBweCIgaGVpZ2h0PSI0MHB4IiB2aWV3Qm94PSIwIDAgNDAgNDAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDUyLjIgKDY3MTQ1KSAtIGh0dHA6Ly93d3cuYm9oZW1pYW5jb2RpbmcuY29tL3NrZXRjaCAtLT4KICAgIDx0aXRsZT5FeHRlbnNpb25zL1NvZnR3YXJlL1ZpZGVvLVNlbnNpbmctQmxvY2s8L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iRXh0ZW5zaW9ucy9Tb2Z0d2FyZS9WaWRlby1TZW5zaW5nLUJsb2NrIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2Utb3BhY2l0eT0iMC4xNSI+CiAgICAgICAgPGcgaWQ9InZpZGVvLW1vdGlvbiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDEwLjAwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyIgc3Ryb2tlPSIjMDAwMDAwIj4KICAgICAgICAgICAgPGNpcmNsZSBpZD0iT3ZhbC1Db3B5IiBmaWxsPSIjRkZGRkZGIiBvcGFjaXR5PSIwLjI1IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGN4PSIzMiIgY3k9IjE2IiByPSI0LjUiPjwvY2lyY2xlPgogICAgICAgICAgICA8Y2lyY2xlIGlkPSJPdmFsLUNvcHkiIGZpbGw9IiNGRkZGRkYiIG9wYWNpdHk9IjAuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjeD0iMzIiIGN5PSIxMiIgcj0iNC41Ij48L2NpcmNsZT4KICAgICAgICAgICAgPGNpcmNsZSBpZD0iT3ZhbC1Db3B5IiBmaWxsPSIjRkZGRkZGIiBvcGFjaXR5PSIwLjc1IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGN4PSIzMiIgY3k9IjgiIHI9IjQuNSI+PC9jaXJjbGU+CiAgICAgICAgICAgIDxjaXJjbGUgaWQ9Ik92YWwiIGZpbGw9IiNGRkZGRkYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY3g9IjMyIiBjeT0iNCIgcj0iNC41Ij48L2NpcmNsZT4KICAgICAgICAgICAgPHBhdGggZD0iTTIyLjY3MTk0NzcsNC40MTk1NzY0OSBMMTYuNSw4LjQxOTkxMjk4IEwxNi41LDYuMSBDMTYuNSw0LjA4OTc2NDU0IDE0LjkzNzE4MDYsMi41IDEzLDIuNSBMNC4xLDIuNSBDMi4wNzYxNDIzNywyLjUgMC41LDQuMDc2MTQyMzcgMC41LDYuMSBMMC41LDE0IEMwLjUsMTUuOTI3Mzk4NyAyLjA4NDQ5ODM5LDE3LjUxMTg5NzEgNC4xLDE3LjYgTDEzLDE3LjYgQzE0LjkwMTY2MDIsMTcuNiAxNi41LDE1Ljk0NjU0NSAxNi41LDE0IEwxNi41LDExLjcxNjkwNDggTDIyLjc1NzI0NzksMTUuNDcxMjUzNSBMMjIuODUzNTUzNCwxNS41NDY0NDY2IEMyMi44NzM3ODg2LDE1LjU2NjY4MTggMjIuOTUxNTMxLDE1LjYgMjMsMTUuNiBDMjMuMjY2OTg2NSwxNS42IDIzLjUsMTUuMzgyNTIwNyAyMy41LDE1LjEgTDIzLjUsNC44IEMyMy41LDQuODM2NzY1MzggMjMuNDQzODA1OCw0LjcwNTY0NTYzIDIzLjM3MTI1MzUsNC41NTcyNDc4OCBDMjMuMjI1OTA1Niw0LjMxNTAwMTM5IDIyLjk0MTU5MzcsNC4yNTgxMzg5OSAyMi42NzE5NDc3LDQuNDE5NTc2NDkgWiIgaWQ9InZpZGVvXzM3XyIgZmlsbD0iIzRENEQ0RCI+PC9wYXRoPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+';

/**
 * Sensor attribute video sensor block should report.
 * @readonly
 * @enum {string}
 */
const SensingAttribute = {
    /** The amount of motion. */
    MOTION: 'motion',

    /** The direction of the motion. */
    DIRECTION: 'direction'
};

/**
 * Subject video sensor block should report for.
 * @readonly
 * @enum {string}
 */
const SensingSubject = {
    /** The sensor traits of the whole stage. */
    STAGE: 'Stage',

    /** The senosr traits of the area overlapped by this sprite. */
    SPRITE: 'this sprite'
};

/**
 * States the video sensing activity can be set to.
 * @readonly
 * @enum {string}
 */
const VideoState = {
    /** Video turned off. */
    OFF: 'off',

    /** Video turned on with default y axis mirroring. */
    ON: 'on',

    /** Video turned on without default y axis mirroring. */
    ON_FLIPPED: 'on-flipped'
};

const EXTENSION_ID = 'objectDetecting';

/**
 * Class for the motion-related blocks in Scratch 3.0
 * @param {Runtime} runtime - the runtime instantiating this block package.
 * @constructor
 */
class Scratch3objectDetectingBlocks {
    constructor (runtime) {
        /**
         * The runtime instantiating this block package.
         * @type {Runtime}
         */
        this.runtime = runtime;

        this.runtime.registerPeripheralExtension(EXTENSION_ID, this);
        this.runtime.connectPeripheral(EXTENSION_ID, 0);
        this.runtime.emit(this.runtime.constructor.PERIPHERAL_CONNECTED);

        /**
         * A flag to determine if this extension has been installed in a project.
         * It is set to false the first time getInfo is run.
         * @type {boolean}
         */
        this.firstInstall = true;

        if (this.runtime.ioDevices) {
            this.runtime.on(Runtime.PROJECT_LOADED, this.projectStarted.bind(this));
            this.runtime.on(Runtime.PROJECT_RUN_START, this.reset.bind(this));
            this._loop();
        }
    }

    /**
     * After analyzing a frame the amount of milliseconds until another frame
     * is analyzed.
     * @type {number}
     */
    static get INTERVAL () {
        return 33;
    }

    /**
     * Dimensions the video stream is analyzed at after its rendered to the
     * sample canvas.
     * @type {Array.<number>}
     */
    static get DIMENSIONS () {
        return [480, 360];
    }

    /**
     * The key to load & store a target's motion-related state.
     * @type {string}
     */
    static get STATE_KEY () {
        return 'Scratch.objectDetecting';
    }

    /**
     * The default motion-related state, to be used when a target has no existing motion state.
     * @type {MotionState}
     */
    static get DEFAULT_MOTION_STATE () {
        return {
            motionFrameNumber: 0,
            motionAmount: 0,
            motionDirection: 0
        };
    }

    /**
     * The transparency setting of the video preview stored in a value
     * accessible by any object connected to the virtual machine.
     * @type {number}
     */
    get globalVideoTransparency () {
        const stage = this.runtime.getTargetForStage();
        if (stage) {
            return stage.videoTransparency;
        }
        return 50;
    }

    set globalVideoTransparency (transparency) {
        const stage = this.runtime.getTargetForStage();
        if (stage) {
            stage.videoTransparency = transparency;
        }
        return transparency;
    }

    /**
     * The video state of the video preview stored in a value accessible by any
     * object connected to the virtual machine.
     * @type {number}
     */
    get globalVideoState () {
        const stage = this.runtime.getTargetForStage();
        if (stage) {
            return stage.videoState;
        }
        // Though the default value for the stage is normally 'on', we need to default
        // to 'off' here to prevent the video device from briefly activating
        // while waiting for stage targets to be installed that say it should be off
        return VideoState.OFF;
    }

    set globalVideoState (state) {
        const stage = this.runtime.getTargetForStage();
        if (stage) {
            stage.videoState = state;
        }
        return state;
    }

    /**
     * Get the latest values for video transparency and state,
     * and set the video device to use them.
     */
    projectStarted () {
        this.setVideoTransparency({
            TRANSPARENCY: this.globalVideoTransparency
        });
        this.videoToggle({
            VIDEO_STATE: this.globalVideoState
        });
    }

    reset () {
    }

    scan() {
    }

    isConnected() {
        if(this._prediction != null) {
            return true;
        }
        else return false;
    }

    connect() {
    }

    async _loop () {
        while (true) {
            this._frame = this.runtime.ioDevices.video.getFrame({
                format: Video.FORMAT_IMAGE_DATA,
                dimensions: Scratch3objectDetectingBlocks.DIMENSIONS
            });
            //console.log(this._frame);
            
            const time = +new Date();
            
            const estimateThrottleTimeout = (+new Date() - time)/4;
            await new Promise(r => setTimeout(r, estimateThrottleTimeout));
        }
    }

    /**
     * Create data for a menu in scratch-blocks format, consisting of an array
     * of objects with text and value properties. The text is a translated
     * string, and the value is one-indexed.
     * @param {object[]} info - An array of info objects each having a name
     *   property.
     * @return {array} - An array of objects with text and value properties.
     * @private
     */
    _buildMenu (info) {
        return info.map((entry, index) => {
            const obj = {};
            obj.text = entry.name;
            obj.value = entry.value || String(index + 1);
            return obj;
        });
    }

    static get SensingAttribute () {
        return SensingAttribute;
    }

    /**
     * An array of choices of whether a reporter should return the frame's
     * motion amount or direction.
     * @type {object[]}
     * @param {string} name - the translatable name to display in sensor
     *   attribute menu
     * @param {string} value - the serializable value of the attribute
     */
    get ATTRIBUTE_INFO () {
        return [
            {
                name: formatMessage({
                    id: 'videoSensing.motion',
                    default: 'motion',
                    description: 'Attribute for the "video [ATTRIBUTE] on [SUBJECT]" block'
                }),
                value: SensingAttribute.MOTION
            },
            {
                name: formatMessage({
                    id: 'videoSensing.direction',
                    default: 'direction',
                    description: 'Attribute for the "video [ATTRIBUTE] on [SUBJECT]" block'
                }),
                value: SensingAttribute.DIRECTION
            }
        ];
    }

    static get SensingSubject () {
        return SensingSubject;
    }

    /**
     * An array of info about the subject choices.
     * @type {object[]}
     * @param {string} name - the translatable name to display in the subject menu
     * @param {string} value - the serializable value of the subject
     */
    get SUBJECT_INFO () {
        return [
            {
                name: formatMessage({
                    id: 'videoSensing.sprite',
                    default: 'sprite',
                    description: 'Subject for the "video [ATTRIBUTE] on [SUBJECT]" block'
                }),
                value: SensingSubject.SPRITE
            },
            {
                name: formatMessage({
                    id: 'videoSensing.stage',
                    default: 'stage',
                    description: 'Subject for the "video [ATTRIBUTE] on [SUBJECT]" block'
                }),
                value: SensingSubject.STAGE
            }
        ];
    }

    /**
     * States the video sensing activity can be set to.
     * @readonly
     * @enum {string}
     */
    static get VideoState () {
        return VideoState;
    }

    /**
     * An array of info on video state options for the "turn video [STATE]" block.
     * @type {object[]}
     * @param {string} name - the translatable name to display in the video state menu
     * @param {string} value - the serializable value stored in the block
     */
    get VIDEO_STATE_INFO () {
        return [
            {
                name: formatMessage({
                    id: 'videoSensing.off',
                    default: 'off',
                    description: 'Option for the "turn video [STATE]" block'
                }),
                value: VideoState.OFF
            },
            {
                name: formatMessage({
                    id: 'videoSensing.on',
                    default: 'on',
                    description: 'Option for the "turn video [STATE]" block'
                }),
                value: VideoState.ON
            },
            {
                name: formatMessage({
                    id: 'videoSensing.onFlipped',
                    default: 'on flipped',
                    description: 'Option for the "turn video [STATE]" block that causes the video to be flipped' +
                        ' horizontally (reversed as in a mirror)'
                }),
                value: VideoState.ON_FLIPPED
            }
        ];
    }

    get CLASS_INFO () {
        return [
            {
                name: formatMessage({
                    id: 'objectDetecting.person',
                    default: '사람',
                    description: ''
                }),
                value: 'person'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.bicycle',
                    default: '자전거',
                    description: ''
                }),
                value: 'bicycle'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.car',
                    default: '차',
                    description: ''
                }),
                value: 'car'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.motorcycle',
                    default: '오토바이',
                    description: ''
                }),
                value: 'motorcycle'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.airplane',
                    default: '비행기',
                    description: ''
                }),
                value: 'airplane'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.bus',
                    default: '버스',
                    description: ''
                }),
                value: 'bus'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.train',
                    default: '기차',
                    description: ''
                }),
                value: 'train'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.truck',
                    default: '트럭',
                    description: ''
                }),
                value: 'truck'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.traffic_light',
                    default: '신호등',
                    description: ''
                }),
                value: 'traffic light'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.stop_sign',
                    default: '정지 신호',
                    description: ''
                }),
                value: 'stop sign'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.parking_meter',
                    default: '주차료 징수기',
                    description: ''
                }),
                value: 'parking meter'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.bird',
                    default: '새',
                    description: ''
                }),
                value: 'bird'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.cat',
                    default: '고양이',
                    description: ''
                }),
                value: 'cat'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.dog',
                    default: '개',
                    description: ''
                }),
                value: 'dog'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.horse',
                    default: '말',
                    description: ''
                }),
                value: 'horse'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.sheep',
                    default: '양',
                    description: ''
                }),
                value: 'sheep'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.cow',
                    default: '소',
                    description: ''
                }),
                value: 'cow'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.elephant',
                    default: '코끼리',
                    description: ''
                }),
                value: 'elephant'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.bear',
                    default: '곰',
                    description: ''
                }),
                value: 'bear'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.zebra',
                    default: '얼룩말',
                    description: ''
                }),
                value: 'zebra'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.giraffe',
                    default: '기린',
                    description: ''
                }),
                value: 'giraffe'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.backpack',
                    default: '백팩',
                    description: ''
                }),
                value: 'backpack'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.umbrella',
                    default: '우산',
                    description: ''
                }),
                value: 'umbrella'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.handbag',
                    default: '핸드백',
                    description: ''
                }),
                value: 'handbag'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.tie',
                    default: '넥타이',
                    description: ''
                }),
                value: 'tie'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.suitcase',
                    default: '여행가방',
                    description: ''
                }),
                value: 'suitcase'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.frisbee',
                    default: '프리스비',
                    description: ''
                }),
                value: 'frisbee'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.skis',
                    default: '스키',
                    description: ''
                }),
                value: 'skis'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.snowboard',
                    default: '스노우보드',
                    description: ''
                }),
                value: 'snowboard'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.sports_ball',
                    default: '스포츠볼',
                    description: ''
                }),
                value: 'sports ball'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.kite',
                    default: '연',
                    description: ''
                }),
                value: 'kite'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.baseball_glove',
                    default: '야구 글러브',
                    description: ''
                }),
                value: 'baseball glove'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.skateboard',
                    default: '스케이트보드',
                    description: ''
                }),
                value: 'skateboard'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.surfboard',
                    default: '서핑보드',
                    description: ''
                }),
                value: 'surfboard'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.tennis_racket',
                    default: '테니스 라켓',
                    description: ''
                }),
                value: 'tennis racket'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.bottle',
                    default: '병',
                    description: ''
                }),
                value: 'bottle'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.wine_glass',
                    default: '와인잔',
                    description: ''
                }),
                value: 'wine glass'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.cup',
                    default: '컵',
                    description: ''
                }),
                value: 'cup'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.fork',
                    default: '포크',
                    description: ''
                }),
                value: 'fork'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.knife',
                    default: '칼',
                    description: ''
                }),
                value: 'knife'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.spoon',
                    default: '스푼',
                    description: ''
                }),
                value: 'spoon'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.bowl',
                    default: '그릇',
                    description: ''
                }),
                value: 'bowl'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.banana',
                    default: '바나나',
                    description: ''
                }),
                value: 'banana'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.apple',
                    default: '사과',
                    description: ''
                }),
                value: 'apple'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.sandwhich',
                    default: '샌드위치',
                    description: ''
                }),
                value: 'sandwhich'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.orange',
                    default: '오렌지',
                    description: ''
                }),
                value: 'orange'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.broccoli',
                    default: '브로콜리',
                    description: ''
                }),
                value: 'broccoli'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.carrot',
                    default: '당근',
                    description: ''
                }),
                value: 'carrot'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.pizza',
                    default: '피자',
                    description: ''
                }),
                value: 'pizza'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.donut',
                    default: '도넛',
                    description: ''
                }),
                value: 'donut'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.cake',
                    default: '케이크',
                    description: ''
                }),
                value: 'cake'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.chair',
                    default: '의자',
                    description: ''
                }),
                value: 'chair'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.couch',
                    default: '소파',
                    description: ''
                }),
                value: 'couch'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.potted_plant',
                    default: '화분',
                    description: ''
                }),
                value: 'potted plant'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.bed',
                    default: '침대',
                    description: ''
                }),
                value: 'bed'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.dining_table',
                    default: '식사 테이블',
                    description: ''
                }),
                value: 'dining table'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.toilet',
                    default: '화장실',
                    description: ''
                }),
                value: 'toilet'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.tv',
                    default: '티비',
                    description: ''
                }),
                value: 'tv'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.laptop',
                    default: '노트북',
                    description: ''
                }),
                value: 'laptop'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.mouse',
                    default: '마우스',
                    description: ''
                }),
                value: 'mouse'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.remote_keyboard',
                    default: '키보드',
                    description: ''
                }),
                value: 'remote keyboard'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.cell_phone',
                    default: '휴대전화',
                    description: ''
                }),
                value: 'cell phone'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.microwave',
                    default: '전자렌지',
                    description: ''
                }),
                value: 'microwave'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.oven',
                    default: '오븐',
                    description: ''
                }),
                value: 'oven'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.toaster',
                    default: '스노우보드',
                    description: ''
                }),
                value: 'toaster'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.sink',
                    default: '싱크대',
                    description: ''
                }),
                value: 'sink'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.refrigerator',
                    default: '냉장고',
                    description: ''
                }),
                value: 'refrigerator'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.book',
                    default: '책',
                    description: ''
                }),
                value: 'book'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.clock',
                    default: '시계',
                    description: ''
                }),
                value: 'clock'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.vase',
                    default: '꽃병',
                    description: ''
                }),
                value: 'vase'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.scissor',
                    default: '가위',
                    description: ''
                }),
                value: 'scissor'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.teddy_bear',
                    default: '곰 인형',
                    description: ''
                }),
                value: 'teddy bear'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.hair_drier',
                    default: '헤어드라이어',
                    description: ''
                }),
                value: 'hair drier'
            },
            {
                name: formatMessage({
                    id: 'objectDetecting.toothbrush',
                    default: '칫솔',
                    description: ''
                }),
                value: 'toothbrush'
            },
        ];
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
            this.globalVideoState = VideoState.ON;
            this.globalVideoTransparency = 50;
            this.projectStarted();
            this.firstInstall = false;
            this._frame = null;
            this._detectValue= 0.7;
            this._cocoSsdModel = null;
            this._prediction = null;
        }

        // Return extension definition
        return {
            id: EXTENSION_ID,
            name: formatMessage({
                id: 'objectDetecting.categoryName',
                default: '물체 인식',
                description: 'Label for coco-ssd category'
            }),
            showStatusButton: true,
            blockIconURI: blockIconURI,
            menuIconURI: menuIconURI,
            blocks: [
                /**
                {
                    opcode: "repeatHat",
                    text: '[SECOND]초 간격으로 계속 실행',
                    blockType: BlockType.HAT,
                    arguments: {
                        SECOND: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 1,
                        },
                    },
                },
                */
                {
                    opcode: "detectVideo",
                    text: '카메라 이미지 분석',
                    blockType: BlockType.COMMAND,
                    arguments: {
                    },
                },
                /**
                {
                    opcode: "detectImageUrl",
                    text: '이미지 URL[URL] 분석',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        URL: {
                            type: ArgumentType.STRING,
                            defaultValue: "Enter URL",
                        },
                    },
                },
                */
                {
                    opcode: "setDetectValue",
                    text: '인식 임계값을 [VALUE]%으로 설정',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        VALUE: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 50,
                        },
                    },
                },
                {
                    opcode: "returnObjectAmount",
                    text: '인식된 객체의 개수',
                    blockType: BlockType.REPORTER,
                    arguments: {
                    },
                },
                {
                    opcode: "returnObjectName",
                    text: '[ORDER]번쨰 객체의 클래스',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        ORDER: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 1,
                        },
                    },
                },
                {
                    opcode: "returnObjectInfo",
                    text: '[ORDER]번쨰 객체의 [INFO]',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        ORDER: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 1,
                        },
                        INFO: {
                            type: ArgumentType.STRING,
                            menu: "infomenu",
                            defaultValue: "accuracy",
                        },
                    },
                },
                {
                    opcode: "returnClassAmount",
                    text: '인식된 [CLASS]의 개수',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        CLASS: {
                            type: ArgumentType.STRING,
                            menu: "classmenu",
                            defaultValue: "person",
                        }, defaultValue: "accuracy",
                    },
                },
                {
                    opcode: "returnClassNumber",
                    text: '[ORDER]번째 인식된 [CLASS]의 객체 번호',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        ORDER: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 1,
                        },
                        CLASS: {
                            type: ArgumentType.STRING,
                            menu: "classmenu",
                            defaultValue: "person",
                        }, defaultValue: "accuracy",
                    },
                },
                {
                    opcode: "checkObjectExist",
                    text: '[CLASS]이(가) 인식되었다면',
                    blockType: BlockType.BOOLEAN,
                    arguments: {
                        CLASS: {
                            type: ArgumentType.STRING,
                            menu: "classmenu",
                            defaultValue: "person",
                        },
                    },
                },
                '---',
                {
                    opcode: 'videoToggle',
                    text: formatMessage({
                        id: 'videoSensing.videoToggle',
                        default: 'turn video [VIDEO_STATE]',
                        description: 'Controls display of the video preview layer'
                    }),
                    arguments: {
                        VIDEO_STATE: {
                            type: ArgumentType.NUMBER,
                            menu: 'VIDEO_STATE',
                            defaultValue: VideoState.OFF
                        }
                    }
                },
                {
                    opcode: 'setVideoTransparency',
                    text: formatMessage({
                        id: 'videoSensing.setVideoTransparency',
                        default: 'set video transparency to [TRANSPARENCY]',
                        description: 'Controls transparency of the video preview layer'
                    }),
                    arguments: {
                        TRANSPARENCY: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 50
                        }
                    }
                },
            ],
            menus: {
                infomenu: {
                    acceptReporters: true,
                    items: [
                        {text: '정확도', value: 'accuracy'},
                        {text: 'X 위치', value: 'x_position'},
                        {text: 'Y 위치', value: 'y_position'},
                        {text: '너비', value: 'width'},
                        {text: '높이', value: 'height'},
                    ]
                },
                VIDEO_STATE: {
                    acceptReporters: true,
                    items: this._buildMenu(this.VIDEO_STATE_INFO)
                },
                classmenu: {
                    acceptReporters: true,
                    items: this._buildMenu(this.CLASS_INFO)
                }
            }
        };
    }

    async repeatHat(args) {
        /**
        await setTimeout(function () {
            return true;
        }, args.SECOND * 1000);
        */
       return true;
    }

    async detectVideo() {
        if (this._frame) {
            this._prediction = await this.estimateObjectOnImage(this._frame);
            console.log(JSON.stringify(this._prediction, null, 2));
            if (this._prediction.length > 0) {
                this.runtime.emit(this.runtime.constructor.PERIPHERAL_CONNECTED);
            } else {
                this.runtime.emit(this.runtime.constructor.PERIPHERAL_DISCONNECTED);
            }
        }
    }

    async estimateObjectOnImage(imageElement) {
        const cocoSsdModel = await this.ensurecocoSsdModelLoaded();
        const prediction = await cocoSsdModel.detect(imageElement, 2, this._detectValue);
        return prediction;
    }

    async ensurecocoSsdModelLoaded() {
        if (!this._cocoSsdModel) {
            console.log("loading start");
            this._cocoSsdModel = await cocoSsd.load();
            console.log("loading end");
        }
        return this._cocoSsdModel;
    }

    async setDetectValue(args) {
        this._detectValue = args.VALUE / 100;
    }

    async returnObjectAmount() {
        if (this._prediction) {
            return this._prediction.length;
        }
    }

    async returnObjectName(args) {
        if (this._prediction) {
            if (this._prediction[args.ORDER - 1]) {
                console.log(this._prediction[args.ORDER - 1].class);
                return this._prediction[args.ORDER - 1].class;
            }
        }
        return "아직 인식되지 않음";
    }

    async returnObjectInfo(args) {
        console.log(this._prediction);
        if (this._prediction) {
            if (this._prediction[args.ORDER - 1]) {
                if (args.INFO == "accuracy") {
                    return this._prediction[args.ORDER - 1].score;
                } else if (args.INFO == "x_position") {
                    return this._prediction[args.ORDER - 1].bbox[0] - (Video.DIMENSIONS[0] / 2);
                } else if (args.INFO == "y_position") {
                    return (Video.DIMENSIONS[1] / 2) - this._prediction[args.ORDER - 1].bbox[1];
                } else if (args.INFO == "width") {
                    return this._prediction[args.ORDER - 1].bbox[2];
                } else if (args.INFO == "height") {
                    return this._prediction[args.ORDER - 1].bbox[3];
                }
            }
            
        }
        return 0;
    }

    async returnClassAmount(args) {
        let amount = 0;
        if (this._prediction) {
            for(let i = 0; i < this._prediction.length; i++) {
                if (this._prediction[i].class == args.CLASS) {
                    amount++;
                }
            }
            return amount;
        }
        return 0;
    }

    async returnClassNumber(args) {
        let amount = 0;
        if (this._prediction) {
            for(let i = 0; i < this._prediction.length; i++) {
                console.log(this._prediction[i].class);
                if (this._prediction[i].class == args.CLASS) {
                    amount++;
                    if(amount == args.ORDER) {
                        return i + 1;
                    }
                }
            }
        }
        return 0;
    }

    async checkObjectExist(args) {
        let amount = 0;
        if (this._prediction) {
            for(let i = 0; i < this._prediction.length; i++) {
                if (this._prediction[i].class === args.CLASS) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * A scratch command block handle that configures the video state from
     * passed arguments.
     * @param {object} args - the block arguments
     * @param {VideoState} args.VIDEO_STATE - the video state to set the device to
     */
    videoToggle (args) {
        const state = args.VIDEO_STATE;
        this.globalVideoState = state;
        if (state === VideoState.OFF) {
            this.runtime.ioDevices.video.disableVideo();
        } else {
            this.runtime.ioDevices.video.enableVideo();
            // Mirror if state is ON. Do not mirror if state is ON_FLIPPED.
            this.runtime.ioDevices.video.mirror = state === VideoState.ON;
        }
    }

    /**
     * A scratch command block handle that configures the video preview's
     * transparency from passed arguments.
     * @param {object} args - the block arguments
     * @param {number} args.TRANSPARENCY - the transparency to set the video
     *   preview to
     */
    setVideoTransparency (args) {
        const transparency = Cast.toNumber(args.TRANSPARENCY);
        this.globalVideoTransparency = transparency;
        this.runtime.ioDevices.video.setPreviewGhost(transparency);
    }
}

module.exports = Scratch3objectDetectingBlocks;