<template>
    <div class="profile-page">
        <section class="section-hero section-shaped my-0">

            <particles-bg class="background-grad-logo shape shape-skew" type="cobweb" color="#FFFFFF" :num="170"/>

            <div class="container shape-container d-flex align-items-center">
                <div class="col px-0">
                    <div class="row justify-content-center align-items-center">
                            <img src="img/brand/headlines/head-poses.png" style="width: 800px; padding-bottom: 80px;" class="img-fluid">                        
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
                            <face-image :poses="true" :face="faceimg"></face-image>
                            <div style="display: flex;">
                              <face-image-very-small @done="incAngle" :lastGeneratedAngle=lastGeneratedAngle :angle="-200"></face-image-very-small>
                              <face-image-very-small @done="incAngle" :lastGeneratedAngle=lastGeneratedAngle :angle="-150"></face-image-very-small>
                              <face-image-very-small @done="incAngle" :lastGeneratedAngle=lastGeneratedAngle :angle="-100"></face-image-very-small>
                              <face-image-very-small @done="incAngle" :lastGeneratedAngle=lastGeneratedAngle :angle="-50"></face-image-very-small>
                            </div>
                            <div style="display: flex;">
                              <face-image-very-small @done="incAngle" :lastGeneratedAngle=lastGeneratedAngle :angle="0"></face-image-very-small>
                              <face-image-very-small @done="incAngle" :lastGeneratedAngle=lastGeneratedAngle :angle="50"></face-image-very-small>
                              <face-image-very-small @done="incAngle" :lastGeneratedAngle=lastGeneratedAngle :angle="100"></face-image-very-small>
                              <face-image-very-small @done="incAngle" :lastGeneratedAngle=lastGeneratedAngle :angle="150"></face-image-very-small>
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
import FaceImageVerySmall from './components/FaceImageVerySmall.vue';


export default {
  components: { 
      FaceImage,
      FaceImageVerySmall,
  },
  data: function() {
    return{
        description: '',
        descriptionLength: 0,
        micClicked: 0,
        generateClicked: false,
        done: true,
        firstGen: false,
        lastGeneratedAngle: -90
      }
    },

  props:{
      faceimg: String,
  },
  methods: {
    incAngle(angle) {
        this.lastGeneratedAngle = angle + 50;
        console.log(angle)
        }
    },
  mounted(){
    this.$nextTick(() => {
        this.lastGeneratedAngle = -200;
    })
  },
  

  };
</script>
<style>
</style>
