/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
using std::normal_distribution;
using std::discrete_distribution;
using std::default_random_engine;
using std::uniform_real_distribution;
using std::uniform_int_distribution;
using std::rand;

using std::string;
using std::vector;
using std::numeric_limits;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 20;  // TODO: Set the number of particles
  std::default_random_engine gen;
 
  //Gaussian distribution for x , y and heading
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // sample from these normal distributions
  for(int i=0;i<num_particles;++i)
  {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  
  for(int i=0;i<num_particles;++i)
  {
    double theta_final, delta_x, delta_y;

    if(fabs(yaw_rate)<0.0001)
    {
      theta_final = particles[i].theta ;
      delta_x = (velocity)*(cos(particles[i].theta)) * delta_t;
      delta_y = (velocity)*(sin(particles[i].theta)) * delta_t;
    }
    else
    {
      theta_final = particles[i].theta + (yaw_rate * delta_t);
      delta_x = (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate * delta_t))-sin(particles[i].theta));
      delta_y = (velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta + (yaw_rate * delta_t)));
    }

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    particles[i].x = particles[i].x + delta_x + dist_x(gen);
    particles[i].y = particles[i].y + delta_y + dist_y(gen);
    particles[i].theta= theta_final + dist_theta(gen);
    
    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(uint16_t i=0;i<observations.size();++i)
  {
    double dist_obs;
    double min_dist = numeric_limits<double>::max();
    int min_id;
    
    for(uint16_t j=0;j<predicted.size();j++)
    {
      dist_obs = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(dist_obs<min_dist)
      {
        min_dist = dist_obs;
        min_id = predicted[j].id;
      }
    }
    observations[i].id = min_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for(int i=0;i<num_particles;++i)
  {
    vector<LandmarkObs> observations_map;
    vector<LandmarkObs> predictions;
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    
    for(uint16_t j=0;j<map_landmarks.landmark_list.size();++j)
    {
      LandmarkObs landmark;
      landmark.id = map_landmarks.landmark_list[j].id_i;
      landmark.x = map_landmarks.landmark_list[j].x_f;
      landmark.y = map_landmarks.landmark_list[j].y_f;
      
      //check if within range of sensor
      if((fabs(particles[i].x -landmark.x)<=sensor_range)&&(fabs(particles[i].y -landmark.y)<=sensor_range))
      {
        predictions.push_back(landmark);
      }
    }
    
    for(uint16_t j=0;j<observations.size();++j)
    {
      double theta = particles[i].theta;
      LandmarkObs observation_map;
      observation_map.id = observations[j].id;
      observation_map.x = particles[i].x + (cos(theta) * observations[j].x) -(sin(theta) * observations[j].y);
      observation_map.y = particles[i].y + (sin(theta) * observations[j].x) + (cos(theta) * observations[j].y);
      observations_map.push_back(observation_map);
    }
    
    //find the associated landmarks and save it inside observations_map
    dataAssociation(predictions,observations_map);
    
    particles[i].weight=1;
    for(uint16_t k=0;k<observations_map.size();++k)
    {
      double sig_x, sig_y, x_obs, y_obs, mu_x, mu_y;
      double gauss_norm,exponent;
      // take the data for associations
      associations.push_back(observations_map[k].id);
      sense_x.push_back(observations_map[k].x);
      sense_y.push_back(observations_map[k].y);
      mu_x = 0;
      mu_y = 0;
      
      // calculate the weights using multivariate gaussian
      for(uint16_t j=0;j<predictions.size();++j)
      {
        if(observations_map[k].id == predictions[j].id)
        {
          mu_x = predictions[j].x;
          mu_y = predictions[j].y;
        }
      }
        
      x_obs = observations_map[k].x;
      y_obs = observations_map[k].y;
      sig_x = std_landmark[0];
      sig_y = std_landmark[1];
      
      gauss_norm = 1 / (2 * M_PI * sig_x * sig_y );
      exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
      particles[i].weight = particles[i].weight * (gauss_norm * exp(-exponent));
     
    }
    
    // setting associations for the current particle
    SetAssociations(particles[i],associations,sense_x,sense_y);
  }
    
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::default_random_engine gen;
  vector<double> particle_weights;

  for (int i = 0; i < num_particles; ++i)
  {
    particle_weights.push_back(particles[i].weight);
  }
  double highest_weight = *max_element(particle_weights.begin(),particle_weights.end());
  
 
  //Uniform distribution for weight
  uniform_real_distribution<double> dist_weight(0, 2*highest_weight);
  uniform_int_distribution<int> dist_index(0, num_particles-1);

  vector<Particle> particles_resampled;
  int index;
  index = dist_index(gen);
  double beta = 0;
  
  

  for(int j=0;j<num_particles;++j)
  {
    double random_weight;
    random_weight = dist_weight(gen);
    beta = beta + random_weight;
    while(beta>particle_weights[index])
    {
      beta = beta - particle_weights[index];
      index = (index+1)%num_particles;
    }
    particles_resampled.push_back(particles[index]);
  }
  particles = particles_resampled;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}