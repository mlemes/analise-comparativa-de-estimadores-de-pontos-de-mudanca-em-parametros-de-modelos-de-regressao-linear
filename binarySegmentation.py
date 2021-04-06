import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble
from sklearn.model_selection import GridSearchCV
from copy import deepcopy
class binarySegmentation:
    def splitIndexesErrors(  data
                            , X
                            , y
                            , model
                            , params
                            , obsMin
                            , obsTotal
                            , crossValidation
                           ):
        '''
        Calculates and returns the average of the squared error obtained from
        predictions of the estimated model into segments below and above
        each data point.
        Parameters: data: dataframe 
                        Dataframe containing observations.
                    X: list
                        Covariates columns names.
                    y: str
                        Dependent variable column name.
                    model: sklearn model
                        Model to be fitted.
                    params: dict
                        Parameters to be passed to the model
                    obsMin: int
                        Minimal quantity of observations into a split
        Returns:
            A numpy matrix containing the following informations as columns:
            [   Viable Index: 
                    index of the split. Only splits that respects the obsMin parameter
                    are actually used for estimating models, that's the origin of the
                    'viable' adjective,
                Mean Squared Error: 
                    Sum of the squared residuals divided by the ammount of observations 
                    in the interval,
                Index where the interval begins,
                Index where the interval ends,
                Sum of the squared residuals below the viable index,
                Quantity of observations below the viable index,
                Sum of the squared residuals above the viable index,
                Quantity of observations above the viable index
            ]
        '''
        indexMin = data.index.min()
        indexMax = data.index.max()
        if indexMax - indexMin + 1 < 2*obsMin: 
            # It's not possible to split a segment in two halfs with obsMin or 
            # more if it contains fewer than the 2*obsMin observations.
            return []
        averageSquaredResiduals = np.zeros(shape=((indexMax - obsMin + 2) - (indexMin + obsMin),8))
        for viableIndex in range(indexMin + obsMin, indexMax - obsMin + 2):
            # A viable index is defined as any index i such that the quantity of
            # observations below (excluding) and above (including) that index
            # satisfies the minimal quantity restriction.
            # By doing that there are no unecessary estimations. The performance gain
            # is linear with the size of the dataset and the quantity restriction.
            # Future improvements:
            #       (1) Add K-fold support
            qtdBelow = viableIndex - indexMin
            qtdAbove = indexMax - viableIndex + 1
            XBelow = data[  (data.index >= indexMin)
                          & (data.index < viableIndex)
                         ][X]
            yBelow = data[  (data.index >= indexMin)
                          & (data.index < viableIndex)
                         ][y]
            XAbove = data[  (data.index >= viableIndex)
                          & (data.index <= indexMax)
                         ][X]
            yAbove = data[  (data.index >= viableIndex)
                          & (data.index <= indexMax)
                         ][y]
            paramsBelow = deepcopy(params)
            paramsAbove = deepcopy(params)
            if model == linear_model.Lasso:
                paramsBelow['alpha'] = paramsBelow['alpha']/np.sqrt(qtdBelow/obsTotal)
                paramsAbove['alpha'] = paramsAbove['alpha']/np.sqrt(qtdAbove/obsTotal)
            if model == ensemble.RandomForestRegressor:
                pass
            if not crossValidation:
                estimatedModelBelow = model(**paramsBelow)
                estimatedModelAbove = model(**paramsAbove)
                estimatedModelBelow.fit(X = XBelow, y = yBelow)
                estimatedModelAbove.fit(X = XAbove, y = yAbove)
            else:
                estimatedModelBelow = GridSearchCV(model(), **params)#, paramsBelow, cv = crossValidation)
                estimatedModelAbove = GridSearchCV(model(), **params)#, paramsAbove, cv = crossValidation)
                estimatedModelBelow.fit(X = XBelow, y = yBelow)
                estimatedModelAbove.fit(X = XAbove, y = yAbove)
                estimatedModelBelow = estimatedModelBelow.best_estimator_
                estimatedModelAbove = estimatedModelAbove.best_estimator_
            sumSquaredResidualsBelow = np.square(estimatedModelBelow.predict(XBelow) - yBelow).sum()
            sumSquaredResidualsAbove = np.square(estimatedModelAbove.predict(XAbove) - yAbove).sum()
            averageSquaredResiduals[viableIndex - (indexMin + obsMin)
                                   ] = [  viableIndex
                                        , (sumSquaredResidualsBelow + sumSquaredResidualsAbove)/(indexMax - indexMin + 1)
                                        , indexMin
                                        , indexMax
                                        , sumSquaredResidualsBelow
                                        , qtdBelow
                                        , sumSquaredResidualsAbove
                                        , qtdAbove
                                       ]
        return averageSquaredResiduals

    def binarySegmentationTree(  data
                               , X
                               , y
                               , model
                               , params
                               , obsMin
                               , maxDepth
                               , obsTotal
                               , crossValidation
                               , splitDepth
                              ):
        '''
        Builds a tree applying the binary segmentation
        algorithm in a recursive way.
        Parameters: data: dataframe 
                        Dataframe containing observations.
                    X: list
                        Covariates columns names.
                    y: str
                        Dependent variable column name.
                    model: sklearn model
                        Model to be fitted.
                    params: dict
                        Parameters to be passed to the model
                    obsMin: int
                        Minimal quantity of observations into a split
                    maxDepth:
                        Maximum depth of the tree. This is conceptually
                        different but numerically equal to the maximum
                        ammount of splits.
                    splitDepth:
                        Depth of the actual split, if realized.
        Returns:
            A list representing a tree, with the following structure
            [ int: splitDepth,
              list: binarySegmentation.splitIndexesErrors object defining this split and some information about it,
              list: Next split object aplied to the interval below this split point, recursively
              list: Next split object aplied to the interval above this split point, recursively
            ]
        '''
        if splitDepth > maxDepth: return []
        splits = binarySegmentation.splitIndexesErrors(  data
                                                       , X
                                                       , y
                                                       , model
                                                       , params
                                                       , obsMin
                                                       , obsTotal
                                                       , crossValidation
                                                      )
        if len(splits)==0: return []
        split  = splits[splits[:,1] == splits[:,1].min()][0].tolist()
        return   [  splitDepth
                  , split
                  , binarySegmentation.binarySegmentationTree(  data[data.index <  int(split[0])]
                                           , X
                                           , y
                                           , model
                                           , params
                                           , obsMin
                                           , maxDepth
                                           , obsTotal
                                           , crossValidation
                                           , splitDepth+1
                                          )
                  , binarySegmentation.binarySegmentationTree(  data[data.index >= int(split[0])]
                                           , X
                                           , y
                                           , model
                                           , params
                                           , obsMin
                                           , maxDepth
                                           , obsTotal
                                           , crossValidation
                                           , splitDepth+1
                                          )
                 ]

    def tree2list(inferedTree, outList):
        '''
        Flatten the binary segmentation tree object into a specified list.
        '''
        outList.append([  inferedTree[0]         # Depth
                        , int(inferedTree[1][2]) # Start of the first segment
                        , int(inferedTree[1][0]) # End of the first segment
                        , inferedTree[1][4]      # Sum of the squared error of the first segment
                       ])
        outList.append([  inferedTree[0]         # Depth
                        , int(inferedTree[1][0]) # Start of the second segment
                        , int(inferedTree[1][3]+1) # End of the second segment
                        , inferedTree[1][6]      # Sum of the squared error of the second segment
                       ])
        if inferedTree[2]: binarySegmentation.tree2list(inferedTree[2], outList)
        if inferedTree[3]: binarySegmentation.tree2list(inferedTree[3], outList)
        return

    def allIntervalSegmentations(inferedTree):
        '''
        Returns all the possible segmentations of the
        interval and it's respective sum of the squared 
        errors and segment quantity based on the binary
        segmentation tree.
        '''
        segmentsList=[]
        binarySegmentation.tree2list(inferedTree, segmentsList)
        segmentsArray = np.array(segmentsList)
        segmentsArray = segmentsArray[np.argsort(segmentsArray[:, 1])]
        segmentedIntervalList = []
        def fillSegmentedIntervalList(segment, stackedSegments):
            '''
            Recursive filling of the segmented interval list.
            '''
            adjacentSegments = segmentsArray[segmentsArray[:, 1] == segment[2]]
            if adjacentSegments.size > 0:
                for nextSegment in adjacentSegments:
                    fillSegmentedIntervalList(nextSegment, stackedSegments + [[segment[1], segment[3]]])
            else: segmentedIntervalList.append(np.array(stackedSegments + [[segment[1], segment[3]]]))
        for inicialSegment in segmentsArray[segmentsArray[:,1]==0]:
            fillSegmentedIntervalList(inicialSegment, [])
        return [[len(segmentedInterval), segmentedInterval[:,1].sum(), segmentedInterval] for segmentedInterval in segmentedIntervalList]

    def bestKSegments(  segmentedIntervalList
                      , maxSegments
                     ):
        '''
        Returns the best interval segmentation for each quantity of splits
        up to maxSplits.
        '''
        bestKSplit = []
        for k in range(2,maxSegments+1):
            minError = np.Inf
            kSegments = [segmentedInterval for segmentedInterval in segmentedIntervalList if segmentedInterval[0] == k]
            bestSplit = None
            for segmentedInterval in kSegments:
                if segmentedInterval[1] < minError:
                    minError = segmentedInterval[1]
                    bestSplit = segmentedInterval
            if bestSplit is not None:
                bestKSplit.append(bestSplit)
        return bestKSplit

    def bestSegment(  bestSegments
                    , segmentPenalty
                   ):
        '''
        Returns the segment that minimizes the definded objetive
        function.
        '''
        minError = np.Inf
        bestSegment = None
        for segmentsQuantity in bestSegments:
            error = segmentsQuantity[1] + segmentPenalty*segmentsQuantity[0]
            if error < minError:
                minError = error
                bestSegment = segmentsQuantity
        return bestSegment

    def binarySegmentation(  data
                           , X
                           , y
                           , model
                           , params
                           , fraqMinObs
                           , maxSegments
                           , segmentPenalty = None
                           , obsMin = None
                           , maxDepth = None
                           , crossValidation = False
                          ):
        '''
        Implements the binary segmentation algorithm as proposed by
        Florencia Leonardi and Peter Bühlmann.
        '''
        obsTotal = len(data)
        _lambda = np.sqrt(np.log(data.shape[1])/(fraqMinObs*obsTotal))
        if model == linear_model.Lasso:
            params['alpha'] = _lambda
        if segmentPenalty is None:
            segmentPenalty = fraqMinObs*_lambda
        if maxDepth is None:
            maxDepth = maxSegments
        if obsMin is None:
            obsMin = int(obsTotal*fraqMinObs)
        #if segmentPenalty is None:
        #        segmentPenalty =
        if not crossValidation:
            bestSegments = [[  1
                             , np.square( model( **params
                                               ).fit(  X=data[X]
                                                     , y=data[y]
                                                    ).predict(data[X]) - data[y]
                                        ).sum()
                             , None
                           ]]
        else:
            bestSegments = [[  1
                 , np.square( GridSearchCV(model(), **params
                                   ).fit(  X=data[X]
                                         , y=data[y]
                                        ).predict(data[X]) - data[y]
                            ).sum()
                 , None
               ]]
        tree = binarySegmentation.binarySegmentationTree(  data
                                      , X
                                      , y
                                      , model
                                      , params
                                      , obsMin
                                      , maxDepth
                                      , obsTotal
                                      , crossValidation
                                      , splitDepth=1
                                    )
        segmentedIntervalList = binarySegmentation.allIntervalSegmentations(tree)
        bestSegments += binarySegmentation.bestKSegments(segmentedIntervalList, maxSegments)
        #return bestSegments
        optimumSegmentation = binarySegmentation.bestSegment(bestSegments, segmentPenalty)
        return optimumSegmentation