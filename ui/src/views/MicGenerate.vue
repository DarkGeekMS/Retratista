<template>
    <div class="profile-page">
        <section class="section-hero section-shaped my-0">

            <particles-bg class="background-grad-logo shape shape-skew" type="cobweb" color="#FFFFFF" :num="170"/>

            <div class="container shape-container d-flex align-items-center">
                <div class="col px-0">
                    <div class="row justify-content-center align-items-center">
                            <img src="img/brand/headlines/voice-generation.png" style="width: 800px; padding-bottom: 80px;" class="img-fluid">                        
                    </div>
                </div>
            </div>

        </section>
        <section class="section section-skew">
            <div class="container">

                <card shadow class="card-profile mt--300" no-body>
                    <div class="px-4">
                        <div class="row justify-content-center">

                            <!-- image -->
                            <face-image :loading="generateClicked" :face="faceimg"></face-image>
                            
                            <!-- mic icon -->
                            <mic-button @clicked="onClickMic"></mic-button>
                                                        
                            <!-- textarea -->
                            <textarea v-model="description" placeholder="Write the face description here!"></textarea>
                            
                            <div style="display: flex;">
                              <generate-button :doneGeneration="done" @clicked="onClickGenerate"></generate-button>
                              <refine-button :firstGen="firstGen" @clicked="onClickRefine"></refine-button>
                              <pose-button :firstGen="firstGen" @clicked="onClickPose"></pose-button>
                            </div>
                        </div>


                    </div>
                </card>
            </div>
        </section>
    </div>
</template>
<script>
import FaceImage from './components/FaceImage.vue';
import MicButton from './components/MicButton.vue';
import GenerateButton from './components/GenerateButton.vue';
import RefineButton from './components/RefineButton.vue';
import PoseButton from './components/PoseButton.vue';
import axios from 'axios'


export default {
  components: { 
      FaceImage,
      MicButton,
      GenerateButton,
      RefineButton,
      PoseButton,
  },
  data: function() {
    return{
        description: '',
        descriptionLength: 0,
        micClicked: 0,
        generateClicked: false,
        faceimg: '',
        done: true,
        firstGen: false,
        logits: []
      }
    },
  
  created() {
    this.recognition =  new webkitSpeechRecognition();
    this.recognition.continuous = true;
    this.recognition.interimResults = false;
    this.recognition.lang = 'en-US';
    this.recognition.maxAlternatives = 5;

    this.recognition.onresult = event => {
        this.description = this.description + event.results[event.results.length - 1][0].transcript;
    };
  },

  methods: {
    onClickMic (micClicked) {
      if (micClicked == 1){
        this.recognition.start();
      }

      if (micClicked == 0){
        this.recognition.stop();
      }
      this.micClicked = micClicked;
    },

    onClickGenerate (generateClicked) {
      this.generateClicked = generateClicked
      this.done = false;
      axios.post(this.$IP + 'tgenerate', {text: this.description})
        .then(function( response ){
          this.faceimg = response.data.face;
          this.generateClicked = false;
          this.done = true;
          this.firstGen = true;
          this.logits = response.data.values;
          
        }.bind(this));
    },

    onClickRefine () {
      this.$router.push({ name: 'refine', params: {logits: this.logits, faceimg: this.faceimg, from_scratch: false} })
    },
    onClickPose () {
      this.$router.push({ name: 'poses', params: {faceimg: this.faceimg} })
    }

  }
  };
</script>
<style>
</style>
