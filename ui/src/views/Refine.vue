<template>
    <div class="profile-page">
        <section class="section-hero section-shaped my-0">

            <particles-bg class="background-grad-logo shape shape-skew" type="cobweb" color="#FFFFFF" :num="170"/>

            <div class="container shape-container d-flex align-items-center">
                <div class="col px-0">
                    <div class="row justify-content-center align-items-center">
                            <img src="img/brand/headlines/face-refinement.png" style="width: 800px; padding-bottom: 80px;" class="img-fluid">                        
                    </div>
                </div>
            </div>

        </section>
        <section class="section section-skew">
            <div class="container" style="width: 100vw !important; max-width: 100%;">

                    <div class="px-4">
                        
                        <div class="row" style="display: flex; align-items: stretch; margin-top: 40px;">
                            
                            <!-- image -->
                            <face-image-small :face="locfaceimg" :loading="generateClicked" :from_scratch="locfrom_scratch && from_scratch" style="margin-top: -10vw; margin-left: 5%; width: 30%;"></face-image-small>
                            

                            <card shadow class="card-profile" style="margin-top: -10vw; margin-left: 7%; width: 50%" no-body>
                                <div class= "circbutton" style="display: flex; margin-bottom: 4vw; margin-top: -60px">
                                    <generate-button-circ :doneGeneration="done" @clicked="onClickGenerate" style="position: -webkit-sticky; top: 0; position:sticky; margin-left: 10px; margin-right: 10px; "></generate-button-circ>
                                    <pose-button-circ @clicked="onClickPose" style="position: -webkit-sticky; top: 0; position:sticky;"></pose-button-circ>
                                </div>
                                <base-button :disabled="generateClicked" type="outline-success" class="btn-3 generate_button" style="margin-top: -9vw" icon="fa fa-repeat" @click="reload" ></base-button>
                                <div style="overflow: auto; height: 500px;">
                                    

                                    <div style="width: 80%; margin: auto; margin-top: 3vw">
                                        <!-- First Section -->
                                        <div>

                                            <general-features style="font: 5em 'Montserrat';"></general-features>

                                            <!-- Gender: -->
                                            <h1 class="routerhead subheadfont" >Gender</h1>
                                            <base-radio name="male" class="mb-3" v-model="radio.gender">
                                                Male
                                            </base-radio>
                                            <base-radio name="female" class="mb-3" v-model="radio.gender">
                                                Female
                                            </base-radio>

                                            <!-- Age: -->
                                            <h1 class="routerhead subheadfont" >Age</h1>
                                            <base-slider v-model="sliders.age"></base-slider>

                                            <!-- Face Thickness: -->
                                            <h1 class="routerhead subheadfont" >Face Thickness</h1>
                                            <base-slider v-model="sliders.thickness"></base-slider>

                                            <!-- Skin Color (Black-White): -->
                                            <h1 class="routerhead subheadfont" >Skin Color</h1>
                                            <p style="text-align: center">(Black-White)</p>
                                            <base-slider v-model="sliders.skin"></base-slider>

                                            <!-- Cheeks: -->
                                            <h1 class="routerhead subheadfont" >Cheeks</h1>
                                            <p style="text-align: center">(White-Rosy)</p>
                                            <base-slider v-model="sliders.cheeks"></base-slider>

                                        </div>


                                        <!-- Eyes -->
                                        <div>
                                            <eyes style="font: 5em 'Montserrat';"></eyes>

                                            <!-- Color: -->
                                            <h1 class="routerhead subheadfont" >Blue Eyes</h1>
                                            <base-slider v-model="sliders.blue_eyes"></base-slider>
                                            <h1 class="routerhead subheadfont" >Green Eyes</h1>
                                            <base-slider v-model="sliders.green_eyes"></base-slider>
                                            <h1 class="routerhead subheadfont" >Brown Eyes</h1>
                                            <base-slider v-model="sliders.brown_eyes"></base-slider>


                                            <!-- Width: -->
                                            <h1 class="routerhead subheadfont" >Eye Width</h1>
                                            <base-slider v-model="sliders.eye_width"></base-slider>

                                            <!-- Eyebrows Thickness: -->
                                            <h1 class="routerhead subheadfont" >Eyebrows Thickness</h1>
                                            <base-slider v-model="sliders.eyebrows_thickness"></base-slider>
                                            
                                            <!-- Bags Under Eyes : -->
                                            <h1 class="routerhead subheadfont" >Eye Bags</h1>
                                            <base-radio name="with" class="mb-3" v-model="radio.eye_bags">
                                                With bags under eyes
                                            </base-radio>
                                            <base-radio name="without" class="mb-3" v-model="radio.eye_bags">
                                                Without bags under eyes
                                            </base-radio>


                                        </div>


                                        <!-- Nose -->
                                        <div>
                                            <nose style="font: 5em 'Montserrat';"></nose>

                                            <!-- Nose Size: -->
                                            <h1 class="routerhead subheadfont" >Nose Size</h1>
                                            <base-slider v-model="sliders.nose_size"></base-slider>
                                    
                                            <!-- Pointy Nose: -->
                                            <h1 class="routerhead subheadfont" >Pointy Nose</h1>
                                            <base-slider v-model="sliders.pointy_nose"></base-slider>

                                        </div>

                                        <!-- Ears -->
                                        <div>
                                            <ears style="font: 5em 'Montserrat';"></ears>

                                            <!-- Ear Size: -->
                                            <h1 class="routerhead subheadfont" >Ear Size</h1>
                                            <base-slider v-model="sliders.ear_size"></base-slider>
                                    
                                        </div>

                                        <!-- Jaw -->
                                        <div>
                                            <jaw style="font: 5em 'Montserrat';"></jaw>

                                            <!-- Lips Size: -->
                                            <h1 class="routerhead subheadfont" >Lip Size</h1>
                                            <base-slider v-model="sliders.lips_size"></base-slider>

                                            <!-- Cheekbones: -->
                                            <h1 class="routerhead subheadfont" >Cheekbones</h1>
                                            <base-slider v-model="sliders.cheekbones"></base-slider>

                                            <!-- Double Chin: -->
                                            <h1 class="routerhead subheadfont" >Double Chin</h1>
                                            <base-slider v-model="sliders.double_chin"></base-slider>


                                        </div>

                                        <!-- Hair -->
                                        <div>

                                            <hair style="font: 5em 'Montserrat';"></hair>

                                            <!-- Color: -->
                                            <h1 class="routerhead subheadfont" >Blonde Hair</h1>
                                            <base-slider v-model="sliders.blonde_hair"></base-slider>
                                            <h1 class="routerhead subheadfont" >Brown Hair</h1>
                                            <base-slider v-model="sliders.brown_hair"></base-slider>
                                            <h1 class="routerhead subheadfont" >Gray Hair</h1>
                                            <base-slider v-model="sliders.gray_hair"></base-slider>
                                            <h1 class="routerhead subheadfont" >Red Hair</h1>
                                            <base-slider v-model="sliders.red_hair"></base-slider>


                                            <!-- Length: -->
                                            <h1 class="routerhead subheadfont" >Hair Length</h1>
                                            <base-slider v-model="sliders.hair_length"></base-slider>
                                            
                                            <!-- Hair Style: -->
                                            <h1 class="routerhead subheadfont" >Hair Style</h1>
                                            <p style="text-align: center">(Straight-Curly)</p>
                                            <base-slider v-model="sliders.hair_style"></base-slider>

                                            <!-- Receding Hairline: -->
                                            <h1 class="routerhead subheadfont" >Receding Hairline</h1>
                                            <base-slider v-model="sliders.receding_hairline"></base-slider>
                                            

                                            <!-- Baldness: -->
                                            <h1 class="routerhead subheadfont" >Baldness</h1>
                                            <base-radio name="bald" class="mb-3" v-model="radio.bald">
                                                Bald
                                            </base-radio>
                                            <base-radio name="not_bald" class="mb-3" v-model="radio.bald">
                                                Not Bald
                                            </base-radio>

                                            <!-- Bangs: -->
                                            <h1 class="routerhead subheadfont" >Bangs</h1>
                                            <base-slider v-model="sliders.bangs"></base-slider>


                                        </div>

                                        <!-- Facial Hair -->
                                        <div>
                                            <facial-hair style="font: 5em 'Montserrat';"></facial-hair>

                                            <!-- Beard: -->
                                            <h1 class="routerhead subheadfont" >Beard</h1>
                                            <base-slider v-model="sliders.beard"></base-slider>
                                        </div>

                                        <!-- Race -->
                                        <div>
                                            <race style="font: 5em 'Montserrat';"></race>

                                            <base-radio name="asian" class="mb-3" v-model="radio.asian">
                                                Asian
                                            </base-radio>
                                            <base-radio name="not_asian" class="mb-3" v-model="radio.asian">
                                                Not Asian
                                            </base-radio>
                                            
                                        </div>

                                        <!-- Accessories -->
                                        <div>

                                            <accessories style="font: 5em 'Montserrat';"></accessories>

                                            <!-- Glasses:  -->
                                            <h1 class="routerhead subheadfont" >Glasses</h1>
                                            <base-radio name="sight" class="mb-3" v-model="radio.glasses">
                                                Sight Glasses
                                            </base-radio>
                                            <base-radio name="sun" class="mb-3" v-model="radio.glasses">
                                                Sun Glasses
                                            </base-radio>
                                            <base-radio name="none" class="mb-3" v-model="radio.glasses">
                                                None
                                            </base-radio>

                                            <!-- Makeup:  -->
                                            <h1 class="routerhead subheadfont" >Makeup</h1>
                                            <base-slider v-model="sliders.makeup"></base-slider>

                                            <!-- Lipstick:  -->
                                            <h1 class="routerhead subheadfont" >Lipstick</h1>
                                            <base-slider v-model="sliders.lipstick"></base-slider>
                                            
                                        </div>

                                        <!-- Morphs -->
                                        <div v-if="!locfrom_scratch">
                                            <morphs style="font: 5em 'Montserrat';"></morphs>

                                            <!-- Eyes Distance:  -->
                                            <h1 class="routerhead subheadfont" >Eyes Distance</h1>
                                            <base-slider v-model="sliders.eyes_distance"></base-slider>

                                            <!-- Eye Eyebrow Distance:  -->
                                            <h1 class="routerhead subheadfont" >Eye Eyebrow Distance</h1>
                                            <base-slider v-model="sliders.eye_eyebrow_distance"></base-slider>

                                            <!-- Nose Mouth Distance:  -->
                                            <h1 class="routerhead subheadfont" >Nose Mouth Distance</h1>
                                            <base-slider v-model="sliders.nose_mouth_distance"></base-slider>

                                            <!-- Open Eyes:  -->
                                            <h1 class="routerhead subheadfont" >Open Eyes</h1>
                                            <base-slider v-model="sliders.open_eyes"></base-slider>

                                            <!-- Mouth Open:  -->
                                            <h1 class="routerhead subheadfont" >Mouth Open</h1>
                                            <base-slider v-model="sliders.mouth_open"></base-slider>

                                            <!-- Smile:  -->
                                            <h1 class="routerhead subheadfont" >Smile</h1>
                                            <base-slider v-model="sliders.smile"></base-slider>
                                            
                                        </div>                                        
                                    </div>
                                </div>
                            </card>
                        </div>
                    </div>
            </div>
        </section>
    </div>
</template>
<script>
import FaceImageSmall from './components/FaceImageSmall.vue';
import GenerateButton from './components/GenerateButton.vue';
import GenerateButtonCirc from './components/GenerateButtonCirc.vue';
import PoseButtonCirc from './components/PoseButtonCirc.vue';
import Headline from './components/Headline.vue';
import Accessories from './components/subheadlines/Accessories.vue';
import Ears from './components/subheadlines/Ears.vue';
import Eyes from './components/subheadlines/Eyes.vue';
import FacialHair from './components/subheadlines/FacialHair.vue';
import GeneralFeatures from './components/subheadlines/GeneralFeatures.vue';
import Hair from './components/subheadlines/Hair.vue';
import Jaw from './components/subheadlines/Jaw.vue';
import Nose from './components/subheadlines/Nose.vue';
import Race from './components/subheadlines/Race.vue';
import Morphs from './components/subheadlines/Morphs.vue';
import axios from 'axios'


export default {
  components: { 
      FaceImageSmall,
      GenerateButton,
      GenerateButtonCirc,
      PoseButtonCirc,
      Headline,
      Accessories,
      Ears,
      Eyes,
      FacialHair,
      GeneralFeatures,
      Hair,
      Jaw,
      Nose,
      Race,
      Morphs,

  },
  data: function() {
    return{
        loclogits: [],
        locfaceimg: '',
        locfrom_scratch: true,
        generateClicked: false,
        done: true,
        sliders: {
            age: 0,
            thickness: 0,
            skin: 0,
            eye_width: 0,
            eyebrows_thickness: 0,
            nose_size: 0,
            ear_size: 0,
            hair_length: 0,
            cheeks:0,
            lips_size: 0,
            cheekbones: 0,
            double_chin: 0,
            hair_style: 0,
            receding_hairline: 0,
            bangs: 0,
            beard: 0,
            lipstick: 0,
            makeup: 0,
            eyes_distance: 0,
            eye_eyebrow_distance: 0,
            nose_mouth_distance: 0,
            open_eyes: 0,
            mouth_open: 0,
            smile: 0,
            red_hair: 0,
            gray_hair: 0,
            blonde_hair: 0,
            brown_hair: 0,
            blue_eyes: 0,
            green_eyes: 0,
            brown_eyes: 0,
            pointy_nose: 0,
            slider2: [150, 400]
        },
        radio: {
            gender: "male",
            eye_bags: "with",
            bald: "not_bald", 
            asian: "not_asian",
            glasses: "none",
        },
        old_features:{
            gender: "male",
            eye_bags: "with",
            bald: "not_bald", 
            asian: "not_asian",
            glasses: "none",
        },
      }
    },

  methods: {
    onClickGenerate (generateClicked) {
      this.generateClicked = generateClicked
      this.done = false;

      if (!this.locfrom_scratch){
        var new_logits = this.attributesToLogits()
        axios.post(this.$IP + 'refine', {values: new_logits})
            .then(function( response ){
                // Handle success
            this.locfaceimg = response.data.face;
            this.generateClicked = false;
            this.done = true;
            this.loclogits = response.data.values;
            this.logitsToAttributes()
            this.locfrom_scratch = false;
            
            }.bind(this));
        }
    else{
        this.reload();
    }
      
    },
    reload(){
        this.generateClicked = true
        this.done = false;

        var new_logits = [];
            for(var i = 0; i < 38; i++) {
                new_logits.push(-1);
            }

        axios.post(this.$IP + 'vgenerate', {values: new_logits})
            .then(function( response ){
                // Handle success
            this.locfaceimg = response.data.face;
            this.generateClicked = false;
            this.done = true;
            this.loclogits = response.data.values;
            this.logitsToAttributes()
            this.locfrom_scratch = false;
            
            }.bind(this));
        
    },
    onClickPose (){
        this.$router.push({ name: 'poses', params: {faceimg: this.locfaceimg} })
    },
    logitsToAttributes (){
        this.sliders.eyebrows_thickness = this.loclogits[0]*100
        this.sliders.red_hair = this.loclogits[1]*100
        this.sliders.blonde_hair = this.loclogits[2]*100
        this.sliders.brown_hair = this.loclogits[3]*100
        this.sliders.gray_hair = this.loclogits[4]*100
        this.sliders.hair_style = this.loclogits[5]*100
        this.sliders.receding_hairline = this.loclogits[6]*100
        if (this.loclogits[7] < 0.5){
            this.radio.bald = 'not_bald'
            this.old_features.bald = 'not_bald'
        }
        else {
            this.radio.bald = 'bald'
            this.old_features.bald = 'bald'
        }
        this.sliders.bangs = this.loclogits[8]*100
        this.sliders.hair_length = this.loclogits[9]*100
        this.sliders.beard = this.loclogits[10]*100
        this.sliders.skin = this.loclogits[11]*100
        if (this.loclogits[12] < 0.8){
            this.radio.asian = 'not_asian'
            this.old_features.asian = 'not_asian'
        }
        else {
            this.radio.asian = 'asian'
            this.old_features.asian = 'asian'
        }
        this.sliders.thickness = this.loclogits[13]*100
        if (this.loclogits[14] < 0.5){
            this.radio.gender = 'female'
            this.old_features.gender = 'female'
        }
        else {
            this.radio.gender = 'male'
            this.old_features.gender = 'male'
        }
        this.sliders.age = this.loclogits[15]*100
        if (this.loclogits[16] < 0.5){
            this.radio.eye_bags = 'without'
            this.old_features.eye_bags = 'without'
        }
        else {
            this.radio.eye_bags = 'with'
            this.old_features.eye_bags = 'with'
        }

        this.sliders.eye_width = this.loclogits[17]*100
        this.sliders.green_eyes = this.loclogits[18]*100
        this.sliders.blue_eyes = this.loclogits[19]*100
        this.sliders.brown_eyes = this.loclogits[20]*100
        this.sliders.lips_size = this.loclogits[21]*100
        this.sliders.nose_size = this.loclogits[22]*100
        this.sliders.ear_size = this.loclogits[23]*100
        this.sliders.double_chin = this.loclogits[24]*100
        this.sliders.cheekbones = this.loclogits[25]*100
        this.sliders.pointy_nose = this.loclogits[26]*100
        this.sliders.cheeks = this.loclogits[27]*100
        this.sliders.makeup = this.loclogits[28]*100
        this.sliders.lipstick = this.loclogits[29]*100;
        if (this.loclogits[30] < 0.6 && this.loclogits[31] < 0.6){
            this.radio.glasses = 'none'
            this.old_features.glasses = 'none'
        }
        else if (Math.abs(this.loclogits[31] - this.getArrayMax([this.loclogits[31], this.loclogits[30]])) < 0.001){
            this.radio.glasses = 'sun'
            this.old_features.glasses = 'sun'
        }
        else if (Math.abs(this.loclogits[30] - this.getArrayMax([this.loclogits[31], this.loclogits[30]])) < 0.001){
            this.radio.glasses = 'sight'
            this.old_features.glasses = 'sight'
        }
        this.sliders.eyes_distance = this.loclogits[32]*100;
        this.sliders.eye_eyebrow_distance = this.loclogits[33]*100;
        this.sliders.nose_mouth_distance = this.loclogits[34]*100;
        this.sliders.open_eyes = this.loclogits[35]*100;
        this.sliders.mouth_open = this.loclogits[36]*100;
        this.sliders.smile = this.loclogits[37]*100;

    },
    attributesToLogits(){
        var new_logits = []

        new_logits.push(this.sliders.eyebrows_thickness/100)
        new_logits.push(this.sliders.red_hair/100)
        new_logits.push(this.sliders.blonde_hair/100)
        new_logits.push(this.sliders.brown_hair/100)
        new_logits.push(this.sliders.gray_hair/100)
        new_logits.push(this.sliders.hair_style/100)
        new_logits.push(this.sliders.receding_hairline/100)



        if(this.old_features.bald != this.radio.bald){

            if(this.radio.bald === 'not_bald'){
                new_logits.push(0)
            }
            else{
                new_logits.push(1)
            }

        }
        else{
            new_logits.push(-1)
        }

        new_logits.push(this.sliders.bangs/100)
        new_logits.push(this.sliders.hair_length/100)
        new_logits.push(this.sliders.beard/100)
        new_logits.push(this.sliders.skin/100)




        if(this.old_features.asian != this.radio.asian){

            if(this.radio.asian === 'not_asian'){
                new_logits.push(0)
            }
            else{
                new_logits.push(1)
            }

        }
        else{
            new_logits.push(-1)
        }

        new_logits.push(this.sliders.thickness/100)

        if(this.old_features.gender != this.radio.gender){

            if(this.radio.gender === 'female'){
                new_logits.push(0)
            }
            else{
                new_logits.push(1)
            }

        }
        else{
            new_logits.push(-1)
        }

        new_logits.push(this.sliders.age/100)

        if(this.old_features.eye_bags != this.radio.eye_bags){

            if(this.radio.eye_bags === 'without'){
                new_logits.push(0)
            }
            else{
                new_logits.push(1)
            }

        }
        else{
            new_logits.push(-1)
        }

        new_logits.push(this.sliders.eye_width/100)
        new_logits.push(this.sliders.green_eyes/100)
        new_logits.push(this.sliders.blue_eyes/100)
        new_logits.push(this.sliders.brown_eyes/100)
        new_logits.push(this.sliders.lips_size/100)
        new_logits.push(this.sliders.nose_size/100)
        new_logits.push(this.sliders.ear_size/100)
        new_logits.push(this.sliders.double_chin/100)
        new_logits.push(this.sliders.cheekbones/100)
        new_logits.push(this.sliders.pointy_nose/100)
        new_logits.push(this.sliders.cheeks/100)
        new_logits.push(this.sliders.makeup/100)
        new_logits.push(this.sliders.lipstick/100)





        if(this.old_features.glasses != this.radio.glasses){

            if(this.radio.glasses === 'sight'){
                new_logits.push(1)
                new_logits.push(0)
            }
            else if(this.radio.glasses === 'sun'){
                new_logits.push(0)
                new_logits.push(1)
            }
            else{
                new_logits.push(0)
                new_logits.push(0)
            }

        }
        else{
            new_logits.push(-1)
            new_logits.push(-1)
        }

        new_logits.push(this.sliders.eyes_distance/100)
        new_logits.push(this.sliders.eye_eyebrow_distance/100)
        new_logits.push(this.sliders.nose_mouth_distance/100)
        if (Math.abs(this.loclogits[17] - (this.sliders.open_eyes/100)) < 0.001){
            new_logits.push(-1)
        }
        else{
            new_logits.push(this.sliders.open_eyes/100)
        }
        // new_logits.push(this.sliders.open_eyes/100)
        new_logits.push(this.sliders.mouth_open/100)
        new_logits.push(this.sliders.smile/100)

        return new_logits;


    },
    getArrayMax(array){
        return Math.max.apply(null, array);
    }
  },
  props:{
      logits: Array,
      faceimg: String,
      from_scratch: {
          type: Boolean,
          default: true
      }
  },
  mounted(){
        this.loclogits = this.logits
        this.locfaceimg = this.faceimg
        this.locfrom_scratch = this.from_scratch
        this.$nextTick(() => {
            
            if(!this.locfrom_scratch){
                this.logitsToAttributes()
            }
        })
    },
  };
</script>
<style>
</style>
