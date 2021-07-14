<template>
    
    <div>
        
        <div class="position-relative">
            
            <!-- shape Hero -->
            <section class="section-shaped  my-0">
                
                <!-- 3D particles background -->
                <particles-bg class="background-grad-5 shape shape-skew" type="cobweb" color="#FFFFFF" :num="170"/>                

                <div class="shape shape-skew">
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <div class="container shape-container d-flex">
                    <div class="col px-0">
                        <div class="row">
                            <div class="col-lg-6">
                                <h1 class="display-3 text-white">You Describe, We Draw!
                                    <span class="text-white">Watch Our Demo Below</span>
                                </h1>
                                <p class="lead text-white">Our Project can help you describe a criminal or a missing person and we will generate photos of them for you as a fast solution to search for them.</p>
                                <div class="btn-wrapper">
                                    <base-button tag="a"
                                                 href="https://github.com/DarkGeekMS/Retratista"
                                                 class="mb-3 mb-sm-0"
                                                 style="background:DodgerBlue; border-style:none"
                                                 icon="fa fa-github">
                                        Star Us on GitHub
                                    </base-button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>


        <!-- Why is Retratista important? -->
        <section  data-aos="fade-up" class="section section-lg">
            <div class="container">
                <div class="row row-grid align-items-center">
                    <div class="order-md-1">
                        <div class="pr-md-5">

                            <div class="d-flex">
                                <div>
                                    <icon name="ni ni-bulb-61" size="lg" class="bg-gradient-white" style="color:DodgerBlue" shadow
                                        rounded></icon>
                                </div>
                                <div class="pl-4 mt-2">
                                    <h4 class="display-3" style="color:DodgerBlue">Why is Retratista important?</h4>
                                </div>
                            </div>

                            <p class="mt-5" align="justify">Sketching a human face from bare description is a cumbersome task, that requires
                                specialization. Many detectives ask for portraitists to sketch a face from description
                                provided by a witness. However, this process has some cons, including :</p>
                            <ul class="list-unstyled mt-3">
                                <li class="py-2">
                                    <div class="d-flex align-items-center">
                                        <div>
                                            <badge style="background:DodgerBlue; color:white" circle class="mr-3" icon="ni ni-fat-delete"></badge>
                                        </div>
                                        <p class="mb-0">It takes hours to sketch a human’s face.</p>
                                    </div>
                                </li>
                                <li class="py-2">
                                    <div class="d-flex align-items-center">
                                        <div>
                                            <badge style="background:DodgerBlue; color:white" circle class="mr-3" icon="ni ni-fat-delete"></badge>
                                        </div>
                                        <p class="mb-0">Any refinements in the face, after being sketched, may require to start sketching
                                        from scratch.</p>
                                    </div>
                                </li>
                                <li class="py-2">
                                    <div class="d-flex align-items-center">
                                        <div>
                                            <badge style="background:DodgerBlue; color:white" circle class="mr-3" icon="ni ni-fat-delete"></badge>
                                        </div>
                                        <p class="mb-0">Sketched faces are not accurate and real enough to be reliable to use in facial
                                        recognition to search for the identity of these faces or any other computer vision
                                        tasks.</p>
                                    </div>
                                </li>
                            </ul>

                            <p align="justify">
                                However, As Artificial Intelligence dominates many fields and Deep Learning models
                                become more and more reliable, we think about leveraging the power of Deep Learning
                                to solve this problem. We propose an application that performs face generation from
                                textual description, as well as providing various refinement and search utilities to identify
                                criminals and missing people, as well.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Statistics -->
        <section  data-aos="fade-up" class="section section-shaped my-0 overflow-hidden">
            <div class="background-grad-6 shape shape-style-3">
                <span></span>
                <span></span>
                <span></span>
                <span></span>
            </div>
            <div class="container pt-sm pb-100">
                <div class="row">
                    <div class="d-flex">
                            <div>
                                <icon name="ni ni-chart-pie-35" size="lg" class="bg-gradient-white" style="color:DodgerBlue" shadow
                                    rounded></icon>
                            </div>
                            <div class="pl-4 mt-2">
                                <h4 class="display-3 text-white">Statistics</h4>
                                <p class="text-white" style = "font-size: 125%;">From International Centre For Missing and Exploited Children.</p>

                            </div>
                    </div>
                </div>
                <div class="row row-grid">


                    <div class="d-flex align-items-center">
                        <icon name="ni ni-fat-delete" class="mr-3" size="sm" gradient="white" shadow round style="color:DodgerBlue"></icon>
                        <p class="text-white mt-3">In USA, an estimated 460,000 children are reported missing every year.</p>
                    </div>

                    <div class="d-flex align-items-center">
                        <icon name="ni ni-fat-delete" class="mr-3" size="sm" gradient="white" shadow round style="color:DodgerBlue"></icon>
                        <p class="text-white mt-3">In UK, an estimated 112,853 children are reported missing every year.</p>
                    </div>

                    <div class="d-flex align-items-center">
                        <icon name="ni ni-fat-delete" class="mr-3" size="sm" gradient="white" shadow round style="color:DodgerBlue"></icon>
                        <p class="text-white mt-3">In Germany, an estimated 100,000 children are reported missing each year.</p>
                    </div>

                    <div class="d-flex align-items-center">
                        <icon name="ni ni-fat-delete" class="mr-3" size="sm" gradient="white" shadow round style="color:DodgerBlue"></icon>
                        <p class="text-white mt-3">In India, an estimated 96,000 children go missing each year.</p>
                    </div>

                    <div class="d-flex align-items-center">
                        <icon name="ni ni-fat-delete" class="mr-3" size="sm" gradient="white" shadow round style="color:DodgerBlue"></icon>
                        <p class="text-white mt-3">In Canada, an estimated 45,288 children are reported missing each year.</p>
                    </div>

                    <div class="d-flex align-items-center">
                        <icon name="ni ni-fat-delete" class="mr-3" size="sm" gradient="white" shadow round style="color:DodgerBlue"></icon>
                        <p class="text-white mt-3">In Russia, an estimated 45,000 children were reported missing in 2015.</p>
                    </div>

                    <div class="d-flex align-items-center">
                        <icon name="ni ni-fat-delete" class="mr-3" size="sm" gradient="white" shadow round style="color:DodgerBlue"></icon>
                        <p class="text-white mt-3">In Australia, an estimated 20,000 children are reported missing every year.</p>
                    </div>
                    
                    <div class="d-flex align-items-center">
                        <icon name="ni ni-fat-delete" class="mr-3" size="sm" gradient="white" shadow round style="color:DodgerBlue"></icon>
                        <p class="text-white mt-3">In Spain, an estimated 20,000 children are reported missing every year.</p>
                    </div>
                        
                </div>
            </div>
        </section>


         <!-- Project Features -->
        <section  data-aos="fade-up" class="section section-lg">
            <div class="container">
                <div class="row row-grid align-items-center">
                    <div class="order-md-1">
                        <div class="pr-md-5">

                            <div class="d-flex">
                                <div>
                                    <icon name="ni ni-diamond" size="lg" class="bg-gradient-white" style="color:DodgerBlue" shadow
                                        rounded></icon>
                                </div>
                                <div class="pl-4 mt-2">
                                    <h4 class="display-3" style="color:DodgerBlue">Project Features</h4>
                                </div>
                            </div>




                            <ul class="list-unstyled mt-5">
                                <li class="py-2">
                                    <div class="d-flex align-items-center">
                                        <div>
                                            <badge style="background:DodgerBlue; color:white" circle class="mr-3" icon="ni ni-fat-delete"></badge>
                                        </div>
                                        <h5 style="color:DodgerBlue">Face Generation</h5>
                                    </div>
                                    <p class="mb-0">The core of the application, where a human face is generated from voice description, textual description or set of toggles and sliders controlling different attributes.</p>
                                </li>
                                <li class="py-2">
                                    <div class="d-flex align-items-center">
                                        <div>
                                            <badge style="background:DodgerBlue; color:white" circle class="mr-3" icon="ni ni-fat-delete"></badge>
                                        </div>
                                        <h5 style="color:DodgerBlue">Face Refinement</h5>
                                    </div>
                                    <p class="mb-0">The helper feature of face generation, where you are
                                    provided by toggles and sliders to refine some facial features of the generated face
                                    .</p>
                                </li>
                                <li class="py-2">
                                    <div class="d-flex align-items-center">
                                        <div>
                                            <badge style="background:DodgerBlue; color:white" circle class="mr-3" icon="ni ni-fat-delete"></badge>
                                        </div>
                                        <h5 style="color:DodgerBlue">Different Head Pose Generation</h5>
                                    </div>
                                    <p class="mb-0">We render the generated face in different
                                    poses, which can be very helpful for both manual and automatic searching of
                                    the described person.</p>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Demo -->
        <section class="section section-shaped background-grad-4 my-0 overflow-hidden" >
            <div class="shape shape-default " style = "background-image: url('img/background/5.jpg'); background-repeat: no-repeat; background-size: cover; background-position: center center;">
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
            </div>
            
            <div class="container py-0">
                <div class="row row-grid align-items-center">
                        <div class="d-flex px-3 mt-5">
                            <div>
                                <icon name="ni ni-button-play" size="lg" class="bg-gradient-white" style="color:DodgerBlue" shadow
                                      rounded></icon>
                            </div>
                            <div class="pl-4 mt-2">
                                <h4 class="display-3 text-white">How To Use?</h4>
                                <p class="text-white" style = "font-size: 125%;">Watch our guide demo for more usage tips.</p>

                                
                                
                            </div>
                            
                        </div>


                        
                    
                </div>
                
            </div>

            <div class="video-button" data-aperture="closed" style="position:relative">
                <div class="video-container" >
                    <video class="actual-video" src="demo.mp4">
                    </video>
                    <div class="play-button__aperture--left"></div>
                    <div class="play-button__aperture--top-right"></div>
                    <div class="play-button__aperture--bottom-right"></div>
                </div>
            </div>

        </section>

        <!-- Sample Results -->
        <section  data-aos="fade-up" class="section section-lg pt-lg-7">
            <div class="container">
                <div class="row justify-content-center">
                    <div class="col-lg-12">


                        <div class="d-flex">
                                <div>
                                    <icon name="ni ni-palette" size="lg" class="bg-gradient-white" style="color:DodgerBlue" shadow
                                        rounded></icon>
                                </div>
                                <div class="pl-4 mt-2">
                                    <h4 class="display-3" style="color:DodgerBlue">Result Samples</h4>
                                </div>
                            </div>


                        <div class="row row-grid mt-5">
                            

                            <!-- result image 1 -->
                            <div class="col-lg-4 ">
                                <card class="border-0 " hover shadow body-classes="py-5">
                                    
                                    <div class="card shadow border-0" style="background:DodgerBlue;">
                                        <img v-lazy="'img/results/1.jpeg'" class="card-img-top">
                                        <blockquote class="card-blockquote">
                                            <svg preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 583 95"
                                                class="svg-bg">
                                                <polygon points="0,52 583,95 0,95" fill="DodgerBlue" ></polygon>
                                                <polygon points="0,42 583,95 683,0 0,95" opacity=".2" fill="DodgerBlue"></polygon>
                                            </svg>
                                            <p class="lead text-italic text-white">woman with lipstick. she has a long hair and rosy cheeks</p>
                                        </blockquote>
                                    </div>

                                </card>
                            </div>

                            <!-- result image 2 -->
                            <div class="col-lg-4">
                                <card class="border-0" hover shadow body-classes="py-5">

                                    <div class="card shadow border-0" style="background:DodgerBlue;">
                                        <img v-lazy="'img/results/2.jpeg'" class="card-img-top">
                                        <blockquote class="card-blockquote">
                                            <svg preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 583 95"
                                                class="svg-bg">
                                                <polygon points="0,52 583,95 0,95" fill="DodgerBlue"></polygon>
                                                <polygon points="0,42 583,95 683,0 0,95" opacity=".2" fill="DodgerBlue"></polygon>
                                            </svg>
                                            <p class="lead text-italic text-white">very old woman with blonde hair and blue eyes</p>
                                        </blockquote>
                                    </div>

                                </card>
                            </div>

                            <!-- result image 3 -->
                            <div class="col-lg-4">
                                <card class="border-0" hover shadow body-classes="py-5">

                                    <div class="card shadow border-0" style="background:DodgerBlue;">
                                        <img v-lazy="'img/results/3.jpeg'" class="card-img-top">
                                        <blockquote class="card-blockquote">
                                            <svg preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 583 95"
                                                class="svg-bg">
                                                <polygon points="0,52 583,95 0,95" fill="DodgerBlue"></polygon>
                                                <polygon points="0,42 583,95 683,0 0,95" opacity=".2" fill="DodgerBlue"></polygon>
                                            </svg>
                                            <p class="lead text-italic text-white">young asian bald man with beard, very wide eyes and glasses</p>
                                        </blockquote>
                                    </div>

                                </card>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>


        <!-- Our Supervisor -->
        <section  data-aos="fade-up" class="section section-lg">
            <div class="container">
                <div class="row justify-content-center text-center mb-sm">
                    <div class="col-lg-8">
                        <h2 class="display-3" style="color:DodgerBlue">Our Supervisor</h2>
                        <p class="lead text-muted">Assistant Professor at Faculty of Engineering, Cairo University.</p>
                    </div>
                </div>
                <div class="row justify-content-center">
                    <div class="col-md-6 col-lg-3 mb-5 mb-lg-0">
                        <div class="px-4">
                            <img v-lazy="'img/team/dr-mayada.jpeg'"
                                 class="rounded-circle img-center img-fluid shadow shadow-lg--hover"
                                 style="width: 200px;">
                            <div class="pt-4 text-center">
                                <h5 class="title">
                                    <span class="d-block mb-1">Dr Mayada Hadhoud</span>
                                </h5>
                                <div class="mt-3">
                                    <!-- <base-button tag="a" href="https://github.com/DarkGeekMS" type="dark" icon="fa fa-github" rounded
                                                 icon-only></base-button> -->
                                    <base-button tag="a" href="https://www.facebook.com/mayada.hadhoud.1" type="dark" icon="fa fa-facebook" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.linkedin.com/in/mayada-hadhoud-b730bb88/" type="dark" icon="fa fa-linkedin" rounded
                                                 icon-only></base-button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Our Team -->
        <section class="section section-sm"  data-aos="fade-up">
            <div class="container">
                <div class="row justify-content-center text-center mb-sm">
                    <div class="col-lg-8">
                        <h2 class="display-3" style="color:DodgerBlue">Our Team</h2>
                        <p class="lead text-muted">We are CMP 2021 Students at Faculty of Engineering, Cairo University.</p>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6 col-lg-3 mb-5 mb-lg-0">
                        <div class="px-4">
                            <img v-lazy="'img/team/shawky.jpeg'"
                                 class="rounded-circle img-center img-fluid shadow shadow-lg--hover"
                                 style="width: 200px;">
                            <div class="pt-4 text-center">
                                <h5 class="title">
                                    <span class="d-block mb-1">Mohamed Shawky</span>
                                </h5>
                                <div class="mt-3">
                                    <base-button tag="a" href="https://github.com/DarkGeekMS" type="warning" icon="fa fa-github" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.facebook.com/mohamed.shawky.7121" type="warning" icon="fa fa-facebook" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.linkedin.com/in/mohamed-shawky/" type="warning" icon="fa fa-linkedin" rounded
                                                 icon-only></base-button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3 mb-5 mb-lg-0">
                        <div class="px-4">
                            <img v-lazy="'img/team/double.jpeg'"
                                 class="rounded-circle img-center img-fluid shadow shadow-lg--hover"
                                 style="width: 200px;">
                            <div class="pt-4 text-center">
                                <h5 class="title">
                                    <span class="d-block mb-1">Mohamed Ahmed</span>
                                </h5>
                                <div class="mt-3">
                                    <base-button tag="a" href="https://github.com/MouhamedAhmed" type="primary" icon="fa fa-github" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.facebook.com/profile.php?id=100013223706744" type="primary" icon="fa fa-facebook" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.linkedin.com/in/mohamedahmed228/" type="primary" icon="fa fa-linkedin" rounded
                                                 icon-only></base-button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3 mb-5 mb-lg-0">
                        <div class="px-4">
                            <img v-lazy="'img/team/monda.jpeg'"
                                 class="rounded-circle img-center img-fluid shadow shadow-lg--hover"
                                 style="width: 200px;">
                            <div class="pt-4 text-center">
                                <h5 class="title">
                                    <span class="d-block mb-1">Remonda Talaat</span>
                                </h5>
                                <div class="mt-3">
                                    <base-button tag="a" href="https://github.com/RemondaTalaat" type="info" icon="fa fa-github" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.facebook.com/remonda.talaat.3" type="info" icon="fa fa-facebook" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.linkedin.com/in/remonda-talaat-206932168/" type="info" icon="fa fa-linkedin" rounded
                                                 icon-only></base-button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3 mb-5 mb-lg-0">
                        <div class="px-4">
                            <img v-lazy="'img/team/ramzy.jpeg'"
                                 class="rounded-circle img-center img-fluid shadow shadow-lg--hover"
                                 style="width: 200px;">
                            <div class="pt-4 text-center">
                                <h5 class="title">
                                    <span class="d-block mb-1">Mohamed Ramzy</span>
                                </h5>
                                <div class="mt-3">
                                    <base-button tag="a" href="https://github.com/MohamedRamzy1" type="success" icon="fa fa-github" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.facebook.com/mohamed.ramzyebaid.1" type="success" icon="fa fa-facebook" rounded
                                                 icon-only></base-button>
                                    <base-button tag="a" href="https://www.linkedin.com/in/mohamed-ramzy-helmy/" type="success" icon="fa fa-linkedin" rounded
                                                 icon-only></base-button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>


    </div>
</template>

<script>

import { ParticlesBg } from "particles-bg-vue";


export default {
  name: "home",
  components: {
      ParticlesBg
  },
  mounted:function(){
    console.clear();

    var videoEl = document.querySelector('video');
    document.querySelector('.video-button').addEventListener('click', 
                                                                function(){
    if(this.dataset.aperture === 'open') {
        this.dataset.aperture = 'closed';
        videoEl.pause();
        videoEl.progress = 0;
    } else {
        this.dataset.aperture = 'open';
        videoEl.play();
    }
    });



  }
  
};
</script>

<style lang="sass">
//set to "white" to hide aperture structure
$aperture-bg: rgba(255, 255, 255, 0.9)
//$aperture-bg: white




.video-button,
.video-container
  position: absolute
  top: 0
  bottom: 0
  left: 0
  right: 0
  margin: auto
  --video-width: 100vw
  width: var(--video-width)
  height: calc(var(--video-width) / 1.77)  

.video-button
  cursor: pointer
  clip-path: polygon(30% 20%, 70% 20%, 70% 80%, 30% 80%)
  transition: 0.6s cubic-bezier(0.6, 0, 0.4, 1)
  &[data-aperture="open"]
    clip-path: polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%)
    
 
.video-container  
  overflow: hidden

.actual-video
  position: absolute
  top: 0
  left: 0
  bottom: 0
  right: 0
  width: 100%
  height: 100%
  transition: 0.6s linear
  transform: scale(1)
  [data-aperture="open"] &
    transform: scale(1.2)
  
@mixin play-button__aperture()
  position: absolute
  background: $aperture-bg
  transition: 0.5s 0.1s cubic-bezier(0.6, 0, 0.4, 1)
  [data-aperture="open"] &
    transition-delay: 0s
  
.play-button__aperture--left
  @include play-button__aperture()
  top: 0
  left: 0
  width: 100%
  height: 100%
  clip-path: polygon(0% 0%, 40% 0%, 40% 100%, 0% 100%)
  [data-aperture="open"] &
    clip-path: polygon(0% 0%, 0% 0%, 0% 100%, 0% 100%)
  
.play-button__aperture--top-right
  @include play-button__aperture()
  top: 0
  left: 0
  width: 100%
  height: 100%
  clip-path: polygon(0% 0%, 100% 0%, 100% 80%)
  [data-aperture="open"] &
    clip-path: polygon(100% 0%, 100% 0%, 100% 0%)
  
.play-button__aperture--bottom-right
  @include play-button__aperture()
  top: 0
  left: 0
  width: 100%
  height: 100%
  clip-path: polygon(0% 100%, 100% 100%, 100% 20%)
  [data-aperture="open"] &
    clip-path: polygon(100% 100%, 100% 100%, 100% 100%)
 
  
</style>
